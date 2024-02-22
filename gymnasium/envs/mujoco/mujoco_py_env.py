from typing import Optional, Union

import gymnasium as gym
from gymnasium import Space, error, logger
from gymnasium.envs.mujoco.mujoco_env_base import BaseMujocoEnv, DEFAULT_SIZE

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "Could not import mujoco_py, which is needed for MuJoCo environments older than V4",
        "You could either use a newer version of the environments, or install the (deprecated) mujoco-py package"
        "following the instructions on their GitHub page."
    ) from e


class MuJocoPyEnv(BaseMujocoEnv):
    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        logger.deprecation(
            "This version of the mujoco environments depends "
            "on the mujoco-py bindings, which are no longer maintained "
            "and may stop working. Please upgrade to the v5 or v4 versions of "
            "the environments (which depend on the mujoco python bindings instead), unless "
            "you are trying to precisely replicate previous works)."
        )

        self.viewer = None
        self._viewers = {}

        super().__init__(
            model_path,
            frame_skip,
            observation_space,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
        )

    def _initialize_simulation(self):
        model = mujoco_py.load_model_from_path(self.fullpath)
        self.sim = mujoco_py.MjSim(model)
        data = self.sim.data
        return model, data

    def _reset_simulation(self):
        self.sim.reset()

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        state = self.sim.get_state()
        state = mujoco_py.MjSimState(state.time, qpos, qvel, state.act, state.udd_state)
        self.sim.set_state(state)
        self.sim.forward()

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            self.sim.step()

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        width, height = self.width, self.height
        camera_name, camera_id = self.camera_name, self.camera_id
        if self.render_mode in {"rgb_array", "depth_array"}:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                if camera_name in self.model._camera_name2id:
                    camera_id = self.model.camera_name2id(camera_name)

                self._get_viewer(self.render_mode).render(
                    width, height, camera_id=camera_id
                )

        if self.render_mode == "rgb_array":
            data = self._get_viewer(self.render_mode).read_pixels(
                width, height, depth=False
            )
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif self.render_mode == "depth_array":
            self._get_viewer(self.render_mode).render(width, height)
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(self.render_mode).read_pixels(
                width, height, depth=True
            )[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif self.render_mode == "human":
            self._get_viewer(self.render_mode).render()

    def _get_viewer(
        self, mode
    ) -> Union["mujoco_py.MjViewer", "mujoco_py.MjRenderContextOffscreen"]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)

            elif mode in {"rgb_array", "depth_array"}:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
            else:
                raise AttributeError(
                    f"Unknown mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer

        return self.viewer

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """
        raise NotImplementedError
