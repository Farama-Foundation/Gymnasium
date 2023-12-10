from os import path

import numpy as np

import gymnasium
from gymnasium.envs.mujoco import MujocoRenderer


try:
    import jax
    import mujoco
    from jax import numpy as jnp
    from mujoco import mjx
except ImportError as e:
    MJX_IMPORT_ERROR = e
else:
    MJX_IMPORT_ERROR = None

DEFAULT_CAMERA_CONFIG = {  # TODO reuse the one from v5
    "distance": 4.0,
}


class MJXEnv(
    gymnasium.functional.FuncEnv[
        mjx._src.types.Data, jnp.ndarray, jnp.ndarray, jnp.ndarray, bool, MujocoRenderer
    ]
):
    """The Base for MJX Environments"""

    def __init__(self, model_path, frame_skip):
        if MJX_IMPORT_ERROR is not None:
            raise gymnasium.error.DependencyNotInstalled(
                f"{MJX_IMPORT_ERROR}. "
                "(HINT: you need to install mujoco, run `pip install gymnasium[mjx]`.)"  # TODO actually create gymnasium[mjx]
            )

        # NOTE can not be JITted because of `Box` not support jax.numpy
        if model_path.startswith(".") or model_path.startswith("/"):  # TODO cleanup
            self.fullpath = model_path
        elif model_path.startswith("~"):
            self.fullpath = path.expanduser(model_path)
        else:
            self.fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        self.frame_skip = frame_skip

        self.model = mujoco.MjModel.from_xml_path(
            self.fullpath
        )  # TODO? do not store and replace with mjx.get_model with mjx==3.1
        # NOTE too much state?
        # alternatives state implementions
        # 1. functional_state = (mjx_data, mjx_model), least internal state in MJXenv, most state in functional_state
        # 2. functional_state = [qpos,qvel], most internal state in MJXenv, least state in functional_state
        self.mjx_model = mjx.device_put(self.model)

        # set action space
        low_action_bound, high_action_bound = self.mjx_model.actuator_ctrlrange.T
        # TODO change bounds and types when and if `Box` supports JAX nativly
        self.action_space = gymnasium.spaces.Box(
            low=np.array(low_action_bound),
            high=np.array(high_action_bound),
            dtype=np.float32,
        )
        # self.action_space = gymnasium.spaces.Box(low=low_action_bound, high=high_action_bound, dtype=low_action_bound.dtype)
        # observation_space: gymnasium.spaces.Box  # set by the sub-class

    def initial(self, rng: jax.random.PRNGKey) -> mjx._src.types.Data:
        mjx_data = mjx.make_data(
            self.model
        )  # TODO? find a more performant alternative that does not allocate?
        qpos, qvel = self._gen_init_state(rng)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        return mjx_data

    def transition(
        self, state: mjx._src.types.Data, action: jnp.ndarray, rng=None
    ) -> mjx._src.types.Data:
        """Step through the simulator using `action` for `self.dt`."""
        mjx_data = state
        mjx_data = mjx_data.replace(ctrl=action)
        mjx_data = jax.lax.fori_loop(
            0, self.frame_skip, lambda _, x: mjx.step(self.mjx_model, x), mjx_data
        )

        return mjx_data
        # TODO fix sensors with MJX>=3.1

    def reward(
        self,
        state: mjx._src.types.Data,
        action: jnp.ndarray,
        next_state: mjx._src.types.Data,
    ) -> jnp.ndarray:
        return self._get_reward(state, action, next_state)[0]

    def transition_info(
        self,
        state: mjx._src.types.Data,
        action: jnp.ndarray,
        next_state: mjx._src.types.Data,
    ) -> dict:
        return self._get_reward(state, action, next_state)[1]

    def render_image(
        self, state: mjx._src.types.Data, render_state: MujocoRenderer
    ) -> tuple[MujocoRenderer, np.ndarray | None]:
        mjx_data = state
        mujoco_renderer = render_state

        data = mujoco.MjData(self.model)
        mjx.device_get_into(data, mjx_data)  # TODO use get_data instead once mjx==3.1
        mujoco.mj_forward(self.model, data)

        mujoco_renderer.data = data

        frame = mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )

        return mujoco_renderer, frame

    def render_init(
        self,
        default_camera_config: dict[str, float] = {},
        camera_id: int | None = None,
        camera_name: str | None = None,
        max_geom=1000,
        width=480,
        height=480,
        render_mode="rgb_array",
    ) -> MujocoRenderer:
        # TODO storing to much state? it should probably be moved internal to MujocoRenderer
        self.render_mode = render_mode
        self.camera_id = camera_id
        self.camera_name = camera_name

        return MujocoRenderer(
            self.model,
            None,
            default_camera_config,
            width,
            height,
            max_geom,
        )

    def render_close(self, render_state: MujocoRenderer) -> None:
        mujoco_renderer = render_state
        if mujoco_renderer is not None:
            mujoco_renderer.close()

    @property
    def dt(self) -> float:
        return self.mjx_model.opt.timestep * self.frame_skip

    def _gen_init_state(self, rng) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns: `(qpos, qvel)`
        """
        # NOTE alternatives
        # 1. return the state in a single vector
        # 2. return it a dictionary keyied by "qpos" & "qvel"
        raise NotImplementedError

    def _get_reward(
        self,
        state: mjx._src.types.Data,
        action: jnp.ndarray,
        next_state: mjx._src.types.Data,
    ) -> tuple[jnp.ndarray, dict]:
        """
        Generates `reward` and `transition_info`, we rely on the JIT's SEE to optimize it.
        Returns: `(reward, transition_info)`
        """
        raise NotImplementedError

    def observation(self, state: mjx._src.types.Data) -> jnp.ndarray:
        raise NotImplementedError

    def terminal(self, state: mjx._src.types.Data) -> bool:
        raise NotImplementedError

    def state_info(self, state: mjx._src.types.Data) -> dict:
        raise NotImplementedError


# TODO in which file to place this class? in `half_cheetah_v5.py`?
class HalfCheetahMJXEnv(MJXEnv, gymnasium.utils.EzPickle):
    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        gymnasium.utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MJXEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=frame_skip,
            **kwargs,
        )

        obs_size = (
            self.mjx_model.nq
            + self.mjx_model.nv
            - exclude_current_positions_from_observation
        )

        self.observation_space = gymnasium.spaces.Box(  # TODO use jnp when and if `Box` supports jax natively
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.mjx_model.nq - 1 * exclude_current_positions_from_observation,
            "qvel": self.mjx_model.nv,
        }

    def _gen_init_state(self, rng) -> tuple[jnp.ndarray, jnp.ndarray]:
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            key=rng, minval=noise_low, maxval=noise_high, shape=(self.mjx_model.nq,)
        )
        qvel = self._reset_noise_scale * jax.random.normal(
            key=rng, shape=(self.mjx_model.nv,)
        )

        return qpos, qvel

    def observation(self, state: mjx._src.types.Data) -> jnp.ndarray:
        mjx_data = state
        position = mjx_data.qpos.flatten()
        velocity = mjx_data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = jnp.concatenate((position, velocity))
        return observation

    def _get_reward(
        self,
        state: mjx._src.types.Data,
        action: jnp.ndarray,
        next_state: mjx._src.types.Data,
    ) -> tuple[jnp.ndarray, dict]:
        mjx_data_old = state
        mjx_data_new = next_state
        x_position_before = mjx_data_old.qpos[0]
        x_position_after = mjx_data_new.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        reward = forward_reward - ctrl_cost
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_velocity": x_velocity,
        }

        return reward, reward_info

    def terminal(self, state: mjx._src.types.Data) -> bool:
        return False
        # NOTE or: return jnp.array(False)

    def state_info(self, state: mjx._src.types.Data) -> dict:
        mjx_data = state
        x_position_after = mjx_data.qpos[0]
        info = {
            "x_position": x_position_after,
        }
        return info

    def render_init(
        self, default_camera_config: dict[str, float] = DEFAULT_CAMERA_CONFIG, **kwargs
    ) -> MujocoRenderer:
        return super().render_init(
            default_camera_config=default_camera_config, **kwargs
        )


# TODO add vector environment
# TODO consider requirement of `metaworld` & `gymansium_robotics.RobotEnv`
