"""Contains the class for the `Swimmer` environment."""

import gymnasium


try:
    import jax
    from jax import numpy as jnp
    from mujoco import mjx
    import flax.struct
    from flax.core.frozen_dict import FrozenDict
except ImportError as e:
    MJX_IMPORT_ERROR = e
else:
    MJX_IMPORT_ERROR = None

from typing import TypedDict
from functools import partial

import numpy as np

from gymnasium.envs.mjx.mjx_env import MJXEnv, _normalize_camera_config
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv
from gymnasium.utils import EzPickle


@flax.struct.dataclass
class SwimmerParams:
    """Parameters for Swimmer environment."""

    xml_file: str
    frame_skip: int
    default_camera_config: FrozenDict[str, float | int | str | None]
    camera_id: int | None
    camera_name: str | None
    max_geom: int
    width: int
    height: int
    render_mode: str | None
    forward_reward_weight: float
    ctrl_cost_weight: float
    reset_noise_scale: float
    exclude_current_positions_from_observation: bool


class Swimmer_MJXEnv(MJXEnv):
    # NOTE: MJX does not yet support condim=1 and therefore this class can not be instantiated
    """Class for Swimmer."""

    def __init__(
        self,
        params: SwimmerParams = None,
    ):
        """Sets the `obveration_space`."""
        if params is None:
            params = self.get_default_params()

        MJXEnv.__init__(self, params=params)

        self.observation_structure = {
            "skipped_qpos": 2 * params.exclude_current_positions_from_observation,
            "qpos": self.mjx_model.nq
            - 2 * params.exclude_current_positions_from_observation,
            "qvel": self.mjx_model.nv,
        }

        obs_size = self.observation_structure["qpos"]
        obs_size += self.observation_structure["qvel"]

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def _gen_init_physics_state(
        self, rng, params: SwimmerParams
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sets `qpos` (positional elements) and `qvel` (velocity elements) form a CUD."""
        noise_low = -params.reset_noise_scale
        noise_high = params.reset_noise_scale

        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            key=rng, minval=noise_low, maxval=noise_high, shape=(self.mjx_model.nq,)
        )
        qvel = jax.random.uniform(
            key=rng, minval=noise_low, maxval=noise_high, shape=(self.mjx_model.nv,)
        )
        act = jnp.empty(self.mjx_model.na)

        return qpos, qvel, act

    def observation(
        self, state: mjx.Data, rng: jax.Array, params: SwimmerParams
    ) -> jnp.ndarray:
        """Observes the `qpos` (posional elements) and `qvel` (velocity elements) of the robot."""
        mjx_data = state

        position = mjx_data.qpos.flatten()
        velocity = mjx_data.qvel.flatten()

        if params.exclude_current_positions_from_observation:
            position = position[2:]

        observation = jnp.concatenate((position, velocity))
        return observation

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: SwimmerParams,
    ) -> tuple[jnp.ndarray, dict]:
        """Reward = reward_dist + reward_ctrl."""
        mjx_data_old = state
        mjx_data_new = next_state

        x_position_before = mjx_data_old.qpos[0]
        x_position_after = mjx_data_new.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt(params)

        forward_reward = params.forward_reward_weight * x_velocity
        ctrl_cost = params.ctrl_cost_weight * jnp.sum(jnp.square(action))

        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def state_info(self, state: mjx.Data, params: SwimmerParams) -> dict[str, float]:
        """Includes state information exclueded from `observation()`."""
        mjx_data = state

        info = {
            "x_position": mjx_data.qpos[0],
            "y_position": mjx_data.qpos[1],
            "distance_from_origin": jnp.linalg.norm(mjx_data.qpos[0:2], ord=2),
        }
        return info

    def get_default_params(self, **kwargs) -> SwimmerParams:
        """Get the default parameter for the Swimmer environment."""
        base = super().get_default_params()
        camera_cfg = kwargs.get("default_camera_config", {})
        camera_cfg = _normalize_camera_config(camera_cfg)

        return SwimmerParams(
            xml_file=kwargs.get("xml_file", "swimmer.xml"),
            frame_skip=kwargs.get("frame_skip", 4),
            default_camera_config=camera_cfg,
            forward_reward_weight=kwargs.get("forward_reward_weight", 1.0),
            ctrl_cost_weight=kwargs.get("ctrl_cost_weight", 1e-4),
            reset_noise_scale=kwargs.get("reset_noise_scale", 0.1),
            exclude_current_positions_from_observation=kwargs.get(
                "exclude_current_positions_from_observation", True
            ),
            camera_id=kwargs.get("camera_id", base.camera_id),
            camera_name=kwargs.get("camera_name", base.camera_name),
            max_geom=kwargs.get("max_geom", base.max_geom),
            width=kwargs.get("width", base.width),
            height=kwargs.get("height", base.height),
            render_mode=kwargs.get("render_mode", base.render_mode),
        )


class SwimmerJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based Swimmer environment using the MJX implementation as base."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs: any):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        temp_env = Swimmer_MJXEnv()
        params = temp_env.get_default_params(**kwargs)

        env = Swimmer_MJXEnv(params=params)
        env.transform(partial(jax.jit, static_argnames="params"))

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=env.metadata,
            render_mode=render_mode,
            kwargs=kwargs,
        )
