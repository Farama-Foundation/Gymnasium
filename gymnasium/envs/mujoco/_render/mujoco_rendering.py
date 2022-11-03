import collections
import weakref
import os
import time
from threading import Lock

import glfw
import imageio
import mujoco
import numpy as np

Contexts = collections.namedtuple('Contexts', ['gl', 'mujoco'])


def _import_egl(width, height):
    from mujoco.egl import GLContext

    return GLContext(width, height)


def _import_glfw(width, height):
    from gymnasium.envs.mujoco._render.glfw_renderer import GLFWContext

    return GLFWContext(width, height)


def _import_osmesa(width, height):
    from mujoco.osmesa import GLContext

    return GLContext(width, height)


_ALL_RENDERERS = collections.OrderedDict(
    [
        ("glfw", _import_glfw),
        ("egl", _import_egl),
        ("osmesa", _import_osmesa),
    ]
)

class MujocoContext:
    """Wrapper for mujoco.MjrContext."""

    def __init__(self,
               model,
               gl_context,
               font_scale=mujoco.mjtFontScale.mjFONTSCALE_150):
        """Initializes this MjrContext instance.

        Args:
        model: An `MjModel` instance.
        gl_context: A `render.ContextBase` instance.
        font_scale: Integer controlling the font size for text. Must be a value
            in `mujoco.mjtFontScale`.

        Raises:
        ValueError: If `font_scale` is invalid.
        """
        if not isinstance(font_scale, mujoco.mjtFontScale):
            font_scale = mujoco.mjtFontScale(font_scale)
        self._gl_context = gl_context
        with gl_context.make_current() as ctx:
            ptr = ctx.call(mujoco.MjrContext, model, font_scale)
            ctx.call(mujoco.mjr_setBuffer, mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ptr)
        gl_context.keep_alive(ptr)
        gl_context.increment_refcount()
        self._ptr = weakref.ref(ptr)

    @property
    def ptr(self):
        return self._ptr()

    def free(self):
        """Frees the native resources held by this MjrContext.

        This is an advanced feature for use when manual memory management is
        necessary. This MjrContext object MUST NOT be used after this function has
        been called.
        """
        if self._gl_context and not self._gl_context.terminated:
            ptr = self.ptr
            if ptr:
                self._gl_context.dont_keep_alive(ptr)
                with self._gl_context.make_current() as ctx:
                    ctx.call(ptr.free)

        if self._gl_context:
            self._gl_context.decrement_refcount()
            self._gl_context.free()
            self._gl_context = None

    def __del__(self):
        self.free()


class RenderContext:
    """Render context superclass for offscreen and window rendering."""

    def __init__(self, model, data, offscreen=True):

        self.model = model
        self.data = data
        self.offscreen = offscreen
        self.offwidth = model.vis.global_.offwidth
        self.offheight = model.vis.global_.offheight
        max_geom = 1000

        mujoco.mj_forward(self.model, self.data)

        self.scn = mujoco.MjvScene(self.model, max_geom)
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._markers = []
        self._overlays = {}

        self._set_mujoco_buffers()

    def _set_mujoco_buffers(self):
        if self.offscreen:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
            if self.con.currentBuffer != mujoco.mjtFramebuffer.mjFB_OFFSCREEN:
                raise RuntimeError("Offscreen rendering not supported")
        else:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con)
            if self.con.currentBuffer != mujoco.mjtFramebuffer.mjFB_WINDOW:
                raise RuntimeError("Window rendering not supported")

    def render(self, camera_id=None, segmentation=False):
        width, height = self.offwidth, self.offheight
        rect = mujoco.MjrRect(left=0, bottom=0, width=width, height=height)

        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        for marker_params in self._markers:
            self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(rect, self.scn, self.con)

        for gridpos, (text1, text2) in self._overlays.items():
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                gridpos,
                rect,
                text1.encode(),
                text2.encode(),
                self.con,
            )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

    def read_pixels(self, depth=True, segmentation=False):
        width, height = self.offwidth, self.offheight
        rect = mujoco.MjrRect(left=0, bottom=0, width=width, height=height)

        rgb_arr = np.zeros(3 * rect.width * rect.height, dtype=np.uint8)
        depth_arr = np.zeros(rect.width * rect.height, dtype=np.float32)

        mujoco.mjr_readPixels(rgb_arr, depth_arr, rect, self.con)
        rgb_img = rgb_arr.reshape(rect.height, rect.width, 3)

        ret_img = rgb_img
        if segmentation:
            seg_img = (
                rgb_img[:, :, 0]
                + rgb_img[:, :, 1] * (2**8)
                + rgb_img[:, :, 2] * (2**16)
            )
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
            seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)

            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]

        if depth:
            depth_img = depth_arr.reshape(rect.height, rect.width)
            return (ret_img, depth_img)
        else:
            return ret_img

    def add_overlay(self, gridpos: int, text1: str, text2: str):
        """Overlays text on the scene."""
        if gridpos not in self._overlays:
            self._overlays[gridpos] = ["", ""]
        self._overlays[gridpos][0] += text1 + "\n"
        self._overlays[gridpos][1] += text2 + "\n"

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError("Ran out of geoms. maxgeom: %d" % self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)
                    )
                )
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self.scn.ngeom += 1

    def close(self):
        """Override close in your rendering subclass to perform any necessary cleanup
        after env.close() is called.
        """
        pass


class RenderContextOffscreen:
    """Offscreen rendering class with opengl context."""
    _contexts = None

    def __new__(cls, *args, **kwargs):
        # TODO(b/174603485): Re-enable once lint stops spuriously firing here.
        obj = super(RenderContextOffscreen, cls).__new__(cls)  # pylint: disable=no-value-for-parameter
        # The lock is created in `__new__` rather than `__init__` because there are
        # a number of existing subclasses that override `__init__` without calling
        # the `__init__` method of the  superclass.
        obj._contexts_lock = Lock()  # pylint: disable=protected-access
        return obj
    
    def __init__(self, model, data):
        # We must make GLContext before MjrContext
        self.model = model
        self.data = data
        self.width = model.vis.global_.offwidth
        self.height = model.vis.global_.offheight

        max_geom = 1000

        self.scn = mujoco.MjvScene(self.model, max_geom)
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()

        self.rect = mujoco.MjrRect(left=0, bottom=0, width=self.width, height=self.height)

        self.pert = mujoco.MjvPerturb()

         # Internal buffers.
        self._rgb_buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._depth_buffer = np.empty((self.height, self.width), dtype=np.float32)

        if self.contexts.mujoco is not None:
            with self.contexts.gl.make_current() as ctx:
                ctx.call(mujoco.mjr_setBuffer, mujoco.mjtFramebuffer.mjFB_OFFSCREEN,
                        self.contexts.mujoco.ptr)

    def _make_rendering_contexts(self):
        """Creates the OpenGL and MuJoCo rendering contexts."""
        # Get the render context
        gl_context = self._get_opengl_backend(self.width, self.height)
        mujoco_context = MujocoContext(self.model, gl_context)
        self._contexts = Contexts(gl=gl_context, mujoco=mujoco_context)
    
    def _free_rendering_contexts(self):
        """Frees existing OpenGL and MuJoCo rendering contexts."""
        self._contexts.mujoco.free()
        self._contexts.gl.free()
        self._contexts = None
        
    @property
    def contexts(self):
        """Returns a `Contexts` namedtuple, used in `Camera`s and rendering code."""
        with self._contexts_lock:
            if not self._contexts:
                self._make_rendering_contexts()
        return self._contexts
    
    def _get_opengl_backend(self, width, height):

        backend = os.environ.get("MUJOCO_GL")
        if backend is not None:
            try:
                return _ALL_RENDERERS[backend](width, height)
            except KeyError:
                raise RuntimeError(
                    "Environment variable {} must be one of {!r}: got {!r}.".format(
                        "MUJOCO_GL", _ALL_RENDERERS.keys(), backend
                    )
                )

        else:
            for name, _ in _ALL_RENDERERS.items():
                try:
                    opengl_context = _ALL_RENDERERS[name](width, height)
                    backend = name
                    return opengl_context
                except:  # noqa:E722
                    pass
            if backend is None:
                raise RuntimeError(
                    "No OpenGL backend could be imported. Attempting to create a "
                    "rendering context will result in a RuntimeError."
                )
            
    def _render_on_gl_thread(self, depth):
        """Performs only those rendering calls that require an OpenGL context."""

        # Render the scene.
        mujoco.mjr_render(self.rect, self.scn,
                        self.contexts.mujoco.ptr)

        # if not depth:
        #     # If rendering RGB, draw any text overlays on top of the image.
        #     for overlay in overlays:
        #         overlay.draw(self._phys.contexts.mujoco.ptr, self._rect)

        # Read the contents of either the RGB or depth buffer.
        mujoco.mjr_readPixels(self._rgb_buffer if not depth else None,
                            self._depth_buffer if depth else None, self.rect,
                            self.contexts.mujoco.ptr)
        
    def render(self, camera_id=None, depth=False, segmentation=False):
        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        with self.contexts.gl.make_current() as ctx:
            ctx.call(self._render_on_gl_thread, depth=depth)

        if depth:
            # Get the distances to the near and far clipping planes.
            extent = self._physics.model.stat.extent
            near = self._physics.model.vis.map.znear * extent
            far = self._physics.model.vis.map.zfar * extent
            # Convert from [0 1] to depth in meters, see links below:
            # http://stackoverflow.com/a/6657284/1461210
            # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
            image = near / (1 - self._depth_buffer * (1 - near / far))
        elif segmentation:
            # Convert 3-channel uint8 to 1-channel uint32.
            image3 = self._rgb_buffer.astype(np.uint32)
            segimage = (image3[:, :, 0] +
                        image3[:, :, 1] * (2**8) +
                        image3[:, :, 2] * (2**16))
            # Remap segid to 2-channel (object ID, object type) pair.
            # Seg ID 0 is background -- will be remapped to (-1, -1).
            segid2output = np.full((self._scene.ngeom + 1, 2), fill_value=-1,
                                    dtype=np.int32)  # Seg id cannot be > ngeom + 1.
            visible_geoms = [g for g in self._scene.geoms if g.segid != -1]
            visible_segids = np.array([g.segid + 1 for g in visible_geoms], np.int32)
            visible_objid = np.array([g.objid for g in visible_geoms], np.int32)
            visible_objtype = np.array([g.objtype for g in visible_geoms], np.int32)
            segid2output[visible_segids, 0] = visible_objid
            segid2output[visible_segids, 1] = visible_objtype
            image = segid2output[segimage]
        else:
            image = self._rgb_buffer

        # The first row in the buffer is the bottom row of pixels in the image.
        return np.flipud(image)


class Viewer:
    """Class for window rendering in all MuJoCo environments."""
    _contexts = None

    def __new__(cls, *args, **kwargs):
        # TODO(b/174603485): Re-enable once lint stops spuriously firing here.
        obj = super(Viewer, cls).__new__(cls)  # pylint: disable=no-value-for-parameter
        # The lock is created in `__new__` rather than `__init__` because there are
        # a number of existing subclasses that override `__init__` without calling
        # the `__init__` method of the  superclass.
        obj._contexts_lock = Lock()  # pylint: disable=protected-access
        return obj

    def __init__(self, model, data):
        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
        self._transparent = False
        self._contacts = False
        self._render_every_frame = True
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._hide_menu = False

        # glfw init
        glfw.init()
        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        self.window = glfw.create_window(width // 2, height // 2, "mujoco", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width

        # set callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        # get viewport
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        super().__init__(model, data, offscreen=False)
        self._init_camera()

        if self.contexts.mujoco is not None:
                with self.contexts.gl.make_current() as ctx:
                    ctx.call(mujoco.mjr_setBuffer, mujoco.mjtFramebuffer.mjFB_OFFSCREEN,
                            self.contexts.mujoco.ptr)

    def _make_rendering_contexts(self):
        """Creates the OpenGL and MuJoCo rendering contexts."""
        # Get the render context
        gl_context = self._get_opengl_backend(self.width, self.height)
        mujoco_context = MujocoContext(self.model, gl_context)
        self._contexts = Contexts(gl=gl_context, mujoco=mujoco_context)
    
    def _free_rendering_contexts(self):
        """Frees existing OpenGL and MuJoCo rendering contexts."""
        self._contexts.mujoco.free()
        self._contexts.gl.free()
        self._contexts = None
        
    @property
    def contexts(self):
        """Returns a `Contexts` namedtuple, used in `Camera`s and rendering code."""
        with self._contexts_lock:
            if not self._contexts:
                self._make_rendering_contexts()
        return self._contexts
    
    
    
    def _init_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        for i in range(3):
            self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
        self.cam.distance = self.model.stat.extent

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return
        # Switch cameras
        elif key == glfw.KEY_TAB:
            self.cam.fixedcamid += 1
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            if self.cam.fixedcamid >= self.model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Pause simulation
        elif key == glfw.KEY_SPACE and self._paused is not None:
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        # Slows down simulation
        elif key == glfw.KEY_S:
            self._run_speed /= 2.0
        # Speeds up simulation
        elif key == glfw.KEY_F:
            self._run_speed *= 2.0
        # Turn off / turn on rendering every frame.
        elif key == glfw.KEY_D:
            self._render_every_frame = not self._render_every_frame
        # Capture screenshot
        elif key == glfw.KEY_T:
            img = np.zeros(
                (
                    glfw.get_framebuffer_size(self.window)[1],
                    glfw.get_framebuffer_size(self.window)[0],
                    3,
                ),
                dtype=np.uint8,
            )
            mujoco.mjr_readPixels(img, None, self.viewport, self.con)
            imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
            self._image_idx += 1
        # Display contact forces
        elif key == glfw.KEY_C:
            self._contacts = not self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
        # Display coordinate frames
        elif key == glfw.KEY_E:
            self.vopt.frame = 1 - self.vopt.frame
        # Hide overlay menu
        elif key == glfw.KEY_H:
            self._hide_menu = not self._hide_menu
        # Make transparent
        elif key == glfw.KEY_R:
            self._transparent = not self._transparent
            if self._transparent:
                self.model.geom_rgba[:, 3] /= 5.0
            else:
                self.model.geom_rgba[:, 3] *= 5.0
        # Geom group visibility
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        # Quit
        if key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            glfw.destroy_window(self.window)
            glfw.terminate()

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self._button_left_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, action, dx / height, dy / height, self.scn, self.cam
            )

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._button_right_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0,
                -0.05 * y_offset,
                self.scn,
                self.cam,
            )

    def _create_overlay(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT

        if self._render_every_frame:
            self.add_overlay(topleft, "", "")
        else:
            self.add_overlay(
                topleft,
                "Run speed = %.3f x real time" % self._run_speed,
                "[S]lower, [F]aster",
            )
        self.add_overlay(
            topleft, "Ren[d]er every frame", "On" if self._render_every_frame else "Off"
        )
        self.add_overlay(
            topleft,
            "Switch camera (#cams = %d)" % (self.model.ncam + 1),
            "[Tab] (camera ID = %d)" % self.cam.fixedcamid,
        )
        self.add_overlay(topleft, "[C]ontact forces", "On" if self._contacts else "Off")
        self.add_overlay(topleft, "T[r]ansparent", "On" if self._transparent else "Off")
        if self._paused is not None:
            if not self._paused:
                self.add_overlay(topleft, "Stop", "[Space]")
            else:
                self.add_overlay(topleft, "Start", "[Space]")
                self.add_overlay(
                    topleft, "Advance simulation by one step", "[right arrow]"
                )
        self.add_overlay(
            topleft, "Referenc[e] frames", "On" if self.vopt.frame == 1 else "Off"
        )
        self.add_overlay(topleft, "[H]ide Menu", "")
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            self.add_overlay(topleft, "Cap[t]ure frame", "Saved as %s" % fname)
        else:
            self.add_overlay(topleft, "Cap[t]ure frame", "")
        self.add_overlay(topleft, "Toggle geomgroup visibility", "0-4")

        self.add_overlay(bottomleft, "FPS", "%d%s" % (1 / self._time_per_render, ""))
        self.add_overlay(
            bottomleft, "Solver iterations", str(self.data.solver_iter + 1)
        )
        self.add_overlay(
            bottomleft, "Step", str(round(self.data.time / self.model.opt.timestep))
        )
        self.add_overlay(bottomleft, "timestep", "%.5f" % self.model.opt.timestep)

    def render(self):
        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.destroy_window(self.window)
                glfw.terminate()
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window
            )
            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    mujoco.MjvPerturb(),
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn,
                )
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(self.viewport, self.scn, self.con)
                # overlay items
                if not self._hide_menu:
                    for gridpos, [t1, t2] in self._overlays.items():
                        mujoco.mjr_overlay(
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            gridpos,
                            self.viewport,
                            t1,
                            t2,
                            self.con,
                        )
                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

            # clear overlay
            self._overlays.clear()

        if self._paused:
            while self._paused:
                update()
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self._run_speed
            )
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # clear markers
        self._markers[:] = []

    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()
