import os
import time

import glfw
import imageio
import mujoco
import numpy as np
from packaging.version import Version

from gymnasium.logger import warn


# The marker API changed in MuJoCo 3.2.0, so we check the mujoco version and set a flag that
# determines which function we use when adding markers to the scene.
_MUJOCO_MARKER_LEGACY_MODE = Version(mujoco.__version__) < Version("3.2.0")


def _import_egl(width, height):
    from mujoco.egl import GLContext

    return GLContext(width, height)


def _import_glfw(width, height):
    from mujoco.glfw import GLContext

    return GLContext(width, height)


def _import_osmesa(width, height):
    from mujoco.osmesa import GLContext

    return GLContext(width, height)


_ALL_RENDERERS = {
    "glfw": _import_glfw,
    "egl": _import_egl,
    "osmesa": _import_osmesa,
}


class BaseRender:
    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        width: int,
        height: int,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
    ):
        """Render context superclass for offscreen and window rendering."""
        self.model = model
        self.data = data

        self._markers = []
        self._overlays = {}

        self.viewport = mujoco.MjrRect(0, 0, width, height)

        # This goes to specific visualizer
        self.scn = mujoco.MjvScene(self.model, max_geom)
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        for flag, value in visual_options.items():
            self.vopt.flags[flag] = value

        self.make_context_current()

        # Keep in Mujoco Context
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._set_mujoco_buffer()

    def _set_mujoco_buffer(self):
        raise NotImplementedError

    def make_context_current(self):
        raise NotImplementedError

    def add_overlay(self, gridpos: int, text1: str, text2: str):
        """Overlays text on the scene."""
        if gridpos not in self._overlays:
            self._overlays[gridpos] = ["", ""]
        self._overlays[gridpos][0] += text1 + "\n"
        self._overlays[gridpos][1] += text2 + "\n"

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker: dict):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(f"Ran out of geoms. maxgeom: {self.scn.maxgeom}")

        if _MUJOCO_MARKER_LEGACY_MODE:  # Old API for markers requires special handling
            self._legacy_add_marker_to_scene(marker)
        else:
            geom_type = marker.get("type", mujoco.mjtGeom.mjGEOM_SPHERE)
            size = marker.get("size", np.array([0.01, 0.01, 0.01]))
            pos = marker.get("pos", np.array([0.0, 0.0, 0.0]))
            mat = marker.get("mat", np.eye(3).flatten())
            rgba = marker.get("rgba", np.array([1.0, 1.0, 1.0, 1.0]))
            mujoco.mjv_initGeom(
                self.scn.geoms[self.scn.ngeom],
                geom_type,
                size=size,
                pos=pos,
                mat=mat,
                rgba=rgba,
            )

        self.scn.ngeom += 1

    def _legacy_add_marker_to_scene(self, marker: dict):
        """Add a marker to the scene compatible with older versions of MuJoCo.

        MuJoCo 3.2 introduced breaking changes to the visual geometries API. To maintain
        compatibility with older versions, we use the legacy API when an older version of MuJoCo is
        detected.

        Args:
            marker: A dictionary containing the marker parameters.
        """
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

    def close(self):
        """Override close in your rendering subclass to perform any necessary cleanup
        after env.close() is called.
        """
        raise NotImplementedError


class OffScreenViewer(BaseRender):
    """Offscreen rendering class with opengl context."""

    def __init__(
        self,
        model: "mujoco.MjMujoco",
        data: "mujoco.MjData",
        width: int,
        height: int,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
    ):
        # We must make GLContext before MjrContext
        self._get_opengl_backend(width, height)

        super().__init__(model, data, width, height, max_geom, visual_options)

        self._init_camera()

    def _init_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        for i in range(3):
            self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
        self.cam.distance = self.model.stat.extent

    def _get_opengl_backend(self, width: int, height: int):
        self.backend = os.environ.get("MUJOCO_GL")
        if self.backend is not None:
            try:
                self.opengl_context = _ALL_RENDERERS[self.backend](width, height)
            except KeyError as e:
                raise RuntimeError(
                    f"Environment variable {'MUJOCO_GL'} must be one of {_ALL_RENDERERS.keys()!r}: got {self.backend!r}."
                ) from e

        else:
            for name, _ in _ALL_RENDERERS.items():
                try:
                    self.opengl_context = _ALL_RENDERERS[name](width, height)
                    self.backend = name
                    break
                except:  # noqa:E722
                    pass
            if self.backend is None:
                raise RuntimeError(
                    "No OpenGL backend could be imported. Attempting to create a "
                    "rendering context will result in a RuntimeError."
                )

    def _set_mujoco_buffer(self):
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

    def make_context_current(self):
        self.opengl_context.make_current()

    def free(self):
        self.opengl_context.free()

    def __del__(self):
        self.free()

    def render(
        self,
        render_mode: str | None,
        camera_id: int | None = None,
        segmentation: bool = False,
    ):
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

        mujoco.mjr_render(self.viewport, self.scn, self.con)

        for gridpos, (text1, text2) in self._overlays.items():
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                gridpos,
                self.viewport,
                text1.encode(),
                text2.encode(),
                self.con,
            )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

        rgb_arr = np.zeros(
            3 * self.viewport.width * self.viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            self.viewport.width * self.viewport.height, dtype=np.float32
        )

        mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.con)

        # Process rendered images according to render_mode
        if render_mode in ["depth_array", "rgbd_tuple"]:
            depth_img = depth_arr.reshape((self.viewport.height, self.viewport.width))
            # original image is upside-down, so flip it
            depth_img = depth_img[::-1, :]
        if render_mode in ["rgb_array", "rgbd_tuple"]:
            rgb_img = rgb_arr.reshape((self.viewport.height, self.viewport.width, 3))
            # original image is upside-down, so flip it
            rgb_img = rgb_img[::-1, :]

            if segmentation:
                seg_img = (
                    rgb_img[:, :, 0]
                    + rgb_img[:, :, 1] * (2**8)
                    + rgb_img[:, :, 2] * (2**16)
                )
                seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
                seg_ids = np.full(
                    (self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32
                )

                for i in range(self.scn.ngeom):
                    geom = self.scn.geoms[i]
                    if geom.segid != -1:
                        seg_ids[geom.segid + 1, 0] = geom.objtype
                        seg_ids[geom.segid + 1, 1] = geom.objid
                rgb_img = seg_ids[seg_img]

        self._markers.clear()
        # Return processed images based on render_mode
        if render_mode == "rgb_array":
            return rgb_img
        elif render_mode == "depth_array":
            return depth_img
        else:  # "rgbd_tuple"
            return rgb_img, depth_img

    def close(self):
        self.free()
        glfw.terminate()


class WindowViewer(BaseRender):
    """Class for window rendering in all MuJoCo environments."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        width: int | None = None,
        height: int | None = None,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
    ):
        glfw.init()

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

        monitor_width, monitor_height = glfw.get_video_mode(
            glfw.get_primary_monitor()
        ).size
        width = monitor_width // 2 if width is None else width
        height = monitor_height // 2 if height is None else height

        glfw.window_hint(glfw.VISIBLE, 1)
        self.window = glfw.create_window(width, height, "mujoco", None, None)

        self.width, self.height = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = self.width * 1.0 / window_width

        # set callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        super().__init__(model, data, width, height, max_geom, visual_options)
        glfw.swap_interval(1)

    def _set_mujoco_buffer(self):
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con)

    def make_context_current(self):
        glfw.make_context_current(self.window)

    def free(self):
        """
        Safely frees the OpenGL context and destroys the GLFW window,
        handling potential issues during interpreter shutdown or resource cleanup.
        """
        try:
            if self.window:
                if glfw.get_current_context() == self.window:
                    glfw.make_context_current(None)
                glfw.destroy_window(self.window)
                self.window = None
        except AttributeError:
            # Handle cases where attributes are missing due to improper environment closure
            warn(
                "Environment was not properly closed using 'env.close()'. Please ensure to close the environment explicitly. "
                "GLFW module or dependencies are unloaded. Window cleanup might not have completed."
            )

    def __del__(self):
        """Eliminate all of the OpenGL glfw contexts and windows"""
        self.free()

    def render(self):
        """
        Renders the environment geometries in the OpenGL glfw window:
            1. Create the overlay for the left side panel menu.
            2. Update the geometries used for rendering based on the current state of the model - `mujoco.mjv_updateScene()`.
            3. Add markers to scene, these are additional geometries to include in the model, i.e arrows, https://mujoco.readthedocs.io/en/latest/APIreference.html?highlight=arrow#mjtgeom.
                These markers are added with the `add_marker()` method before rendering.
            4. Render the 3D scene to the window context - `mujoco.mjr_render()`.
            5. Render overlays in the window context - `mujoco.mjr_overlay()`.
            6. Swap front and back buffer, https://www.glfw.org/docs/3.3/quick.html.
            7. Poll events like mouse clicks or keyboard input.
        """

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

        # clear overlay
        self._overlays.clear()
        # clear markers
        self._markers.clear()

    def close(self):
        self.free()
        glfw.terminate()

    def _key_callback(self, window, key: int, scancode, action: int, mods):
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

    def _cursor_pos_callback(
        self, window: "glfw.LP__GLFWwindow", xpos: float, ypos: float
    ):
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

        mujoco.mjv_moveCamera(
            self.model, action, dx / width, dy / height, self.scn, self.cam
        )

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window: "glfw.LP__GLFWwindow", button, act, mods):
        self._button_left_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._button_right_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

    def _scroll_callback(self, window, x_offset, y_offset: float):
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
        if mujoco.__version__ >= "3.0.0":
            self.add_overlay(
                bottomleft, "Solver iterations", str(self.data.solver_niter[0] + 1)
            )
        elif mujoco.__version__ < "3.0.0":
            self.add_overlay(
                bottomleft, "Solver iterations", str(self.data.solver_iter + 1)
            )
        self.add_overlay(
            bottomleft, "Step", str(round(self.data.time / self.model.opt.timestep))
        )
        self.add_overlay(bottomleft, "timestep", "%.5f" % self.model.opt.timestep)


class MujocoRenderer:
    """This is the MuJoCo renderer manager class for every MuJoCo environment.

    The class has two main public methods available:
    - :meth:`render` - Renders the environment in three possible modes: "human", "rgb_array", or "depth_array"
    - :meth:`close` - Closes all contexts initialized with the renderer

    """

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        default_cam_config: dict | None = None,
        width: int | None = None,
        height: int | None = None,
        max_geom: int = 1000,
        camera_id: int | None = None,
        camera_name: str | None = None,
        visual_options: dict[int, bool] = {},
    ):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            model: MjModel data structure of the MuJoCo simulation
            data: MjData data structure of the MuJoCo simulation
            default_cam_config: dictionary with attribute values of the viewer's default camera, https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global
            width: width of the OpenGL rendering context
            height: height of the OpenGL rendering context
            max_geom: maximum number of geometries to render
            camera_id: The integer camera id from which to render the frame in the MuJoCo simulation
            camera_name: The string name of the camera from which to render the frame in the MuJoCo simulation. This argument should not be passed if using cameara_id instead and vice versa
        """
        self.model = model
        self.data = data
        self._viewers = {}
        self.viewer = None
        self.default_cam_config = default_cam_config
        self.width = width
        self.height = height
        self.max_geom = max_geom
        self._vopt = visual_options

        # set self.camera_id using `camera_id` or `camera_name`
        if camera_id is not None and camera_name is not None:
            raise ValueError(
                "Both `camera_id` and `camera_name` cannot be"
                " specified at the same time."
            )

        no_camera_specified = camera_name is None and camera_id is None
        if no_camera_specified:
            camera_name = "track"

        if camera_id is None:
            self.camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )
        else:
            self.camera_id = camera_id

    def render(
        self,
        render_mode: str | None,
    ):
        """Renders a frame of the simulation in a specific format and camera view.

        Args:
            render_mode: The format to render the frame, it can be: "human", "rgb_array", "depth_array", or "rgbd_tuple"

        Returns:
            If render_mode is "rgb_array" or "depth_array" it returns a numpy array in the specified format. "rgbd_tuple" returns a tuple of numpy arrays of the form (rgb, depth). "human" render mode does not return anything.
        """
        if render_mode != "human":
            assert (
                self.width is not None and self.height is not None
            ), f"The width: {self.width} and height: {self.height} cannot be `None` when the render_mode is not `human`."

        viewer = self._get_viewer(render_mode=render_mode)

        if render_mode in ["rgb_array", "depth_array", "rgbd_tuple"]:
            return viewer.render(render_mode=render_mode, camera_id=self.camera_id)
        elif render_mode == "human":
            return viewer.render()

    def _get_viewer(self, render_mode: str | None):
        """Initializes and returns a viewer class depending on the render_mode
        - `WindowViewer` class for "human" render mode
        - `OffScreenViewer` class for "rgb_array", "depth_array", or "rgbd_tuple" render mode
        """
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            if render_mode == "human":
                self.viewer = WindowViewer(
                    self.model,
                    self.data,
                    self.width,
                    self.height,
                    self.max_geom,
                    self._vopt,
                )
            elif render_mode in {"rgb_array", "depth_array", "rgbd_tuple"}:
                self.viewer = OffScreenViewer(
                    self.model,
                    self.data,
                    self.width,
                    self.height,
                    self.max_geom,
                    self._vopt,
                )
            else:
                raise AttributeError(
                    f"Unexpected mode: {render_mode}, expected modes: human, rgb_array, depth_array, or rgbd_tuple"
                )
            # Add default camera parameters
            self._set_cam_config()
            self._viewers[render_mode] = self.viewer

        if len(self._viewers.keys()) > 1:
            # Only one context can be current at a time
            self.viewer.make_context_current()

        return self.viewer

    def _set_cam_config(self):
        """Set the default camera parameters"""
        assert self.viewer is not None
        if self.default_cam_config is not None:
            for key, value in self.default_cam_config.items():
                if isinstance(value, np.ndarray):
                    getattr(self.viewer.cam, key)[:] = value
                else:
                    setattr(self.viewer.cam, key, value)

    def close(self):
        """Close the OpenGL rendering contexts of all viewer modes"""
        for _, viewer in self._viewers.items():
            viewer.close()
