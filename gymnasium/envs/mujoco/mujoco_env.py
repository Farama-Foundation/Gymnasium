from os import path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, logger, spaces
from gymnasium.spaces import Space


try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None


DEFAULT_SIZE = 480


class BaseMujocoEnv(gym.Env[NDArray[np.float64], NDArray[np.float32]]):
    """Superclass for all MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        """Base abstract class for mujoco based environments.

        Args:
            model_path: Path to the MuJoCo Model.
            frame_skip: Number of MuJoCo simulation steps per gym `step()`.
            observation_space: The observation space of the environment.
            render_mode: The `render_mode` used.
            width: The width of the render window.
            height: The height of the render window.
            camera_id: The camera ID used.
            camera_name: The name of the camera used (can not be used in conjunction with `camera_id`).

        Raises:
            OSError: when the `model_path` does not exist.
            error.DependencyNotInstalled: When `mujoco` is not installed.
        """
        if model_path.startswith(".") or model_path.startswith("/"):
            self.fullpath = model_path
        elif model_path.startswith("~"):
            self.fullpath = path.expanduser(model_path)
        else:
            self.fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        self.width = width
        self.height = height
        # may use width and height
        self.model, self.data = self._initialize_simulation()

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.observation_space = observation_space
        self._set_action_space()

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    # methods to override:
    # ----------------------------
    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]:
        raise NotImplementedError

    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def _initialize_simulation(self) -> Tuple[Any, Any]:
        """
        Initialize MuJoCo simulation data structures mjModel and mjData.
        """
        raise NotImplementedError

    def _reset_simulation(self) -> None:
        """
        Reset MuJoCo simulation data structures, mjModel and mjData.
        """
        raise NotImplementedError

    def _step_mujoco_simulation(self, ctrl, n_frames) -> None:
        """
        Step over the MuJoCo simulation.
        """
        raise NotImplementedError

    def render(self) -> Union[NDArray[np.float64], None]:
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        raise NotImplementedError

    # -----------------------------
    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self._reset_simulation()

        ob = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

    def set_state(self, qpos, qvel) -> None:
        """
        Set the joints position qpos and velocity qvel of the model. Override this method depending on the MuJoCo bindings used.
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)

    def close(self):
        """Close all processes like rendering contexts"""
        raise NotImplementedError

    def get_body_com(self, body_name) -> NDArray[np.float64]:
        """Return the cartesian position of a body frame"""
        raise NotImplementedError

    def state_vector(self) -> NDArray[np.float64]:
        """Return the position and velocity joint states of the model"""
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])


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
        if MUJOCO_PY_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(
                f"{MUJOCO_PY_IMPORT_ERROR}. "
                "(HINT: you need to install mujoco-py, and also perform the setup instructions "
                "here: https://github.com/openai/mujoco-py.)"
            )

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


class MujocoEnv(BaseMujocoEnv):
    """Superclass for MuJoCo environments."""

    def __init__(
        self,
        model_path,
        frame_skip,
        observation_space: Space,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
    ):
        if MUJOCO_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(
                f"{MUJOCO_IMPORT_ERROR}. "
                "(HINT: you need to install mujoco, run `pip install gymnasium[mujoco]`.)"
            )

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

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, default_camera_config, self.width, self.height
        )

    def _initialize_simulation(
        self,
    ) -> Tuple["mujoco._structs.MjModel", "mujoco._structs.MjData"]:
        model = mujoco.MjModel.from_xml_path(self.fullpath)
        # MjrContext will copy model.vis.global_.off* to con.off*
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def render(self):
        return self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos
