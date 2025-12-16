from os import path

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space


try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        'MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`'
    ) from e


DEFAULT_SIZE = 480


def expand_model_path(model_path: str) -> str:
    """Expands the `model_path` to a full path if it starts with '~' or '.' or '/'."""
    if model_path.startswith(".") or model_path.startswith("/"):
        fullpath = model_path
    elif model_path.startswith("~"):
        fullpath = path.expanduser(model_path)
    else:
        fullpath = path.join(path.dirname(__file__), "assets", model_path)
    if not path.exists(fullpath):
        raise OSError(f"File {fullpath} does not exist")

    return fullpath


class MujocoEnv(gym.Env):
    """Superclass for MuJoCo based environments."""

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_space: Space | None,
        render_mode: str | None = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: int | None = None,
        camera_name: str | None = None,
        default_camera_config: dict[str, float | int] | None = None,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
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
            default_camera_config: configuration for rendering camera.
            max_geom: max number of rendered geometries.
            visual_options: render flag options.

        Raises:
            OSError: when the `model_path` does not exist.
            error.DependencyNotInstalled: When `mujoco` is not installed.
        """
        self.fullpath = expand_model_path(model_path)

        self.width = width
        self.height = height
        # may use width and height
        self.model, self.data = self._initialize_simulation()

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip

        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'
        if observation_space is not None:
            self.observation_space = observation_space
        self._set_action_space()

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            default_camera_config,
            self.width,
            self.height,
            max_geom,
            camera_id,
            camera_name,
            visual_options,
        )

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _initialize_simulation(
        self,
    ) -> tuple["mujoco.MjModel", "mujoco.MjData"]:
        """
        Initialize MuJoCo simulation data structures `mjModel` and `mjData`.
        """
        model = mujoco.MjModel.from_xml_path(self.fullpath)
        # MjrContext will copy model.vis.global_.off* to con.off*
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def set_state(self, qpos, qvel):
        """Set the joints position qpos and velocity qvel of the model.

        Note: `qpos` and `qvel` is not the full physics state for all mujoco models/environments https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        """
        Step over the MuJoCo simulation.
        """
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def render(self):
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        """Close rendering contexts processes."""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def get_body_com(self, body_name):
        """Return the cartesian position of a body frame."""
        return self.data.body(body_name).xpos

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

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

    def state_vector(self) -> NDArray[np.float64]:
        """Return the position and velocity joint states of the model.

        Note: `qpos` and `qvel` does not constitute the full physics state for all `mujoco` environments see https://mujoco.readthedocs.io/en/stable/computation/index.html#the-state.
        """
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    # methods to override:
    # ----------------------------
    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        raise NotImplementedError

    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each environment subclass.
        """
        raise NotImplementedError

    def _get_reset_info(self) -> dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

    # -----------------------------
