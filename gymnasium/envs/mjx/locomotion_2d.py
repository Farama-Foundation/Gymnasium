"""Contains the classes for the 2d locomotion environments, `HalfCheetah`, `Hopper` and `Walker2D`."""

import gymnasium


try:
    import jax
    from jax import numpy as jnp
    from mujoco import mjx
except ImportError as e:
    MJX_IMPORT_ERROR = e
else:
    MJX_IMPORT_ERROR = None

from typing import TypedDict

import numpy as np

from gymnasium.envs.mjx.mjx_env import MJXEnv
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco.half_cheetah_v5 import (
    DEFAULT_CAMERA_CONFIG as HALFCHEETAH_DEFAULT_CAMERA_CONFIG,
)
from gymnasium.envs.mujoco.hopper_v5 import (
    DEFAULT_CAMERA_CONFIG as HOPPER_DEFAULT_CAMERA_CONFIG,
)
from gymnasium.envs.mujoco.walker2d_v5 import (
    DEFAULT_CAMERA_CONFIG as WALKER2D_DEFAULT_CAMERA_CONFIG,
)


class Locomotion2dMJXEnvParams(TypedDict):
    """Parameters for the HalfCheetah, Hopper, Walker2d environments."""

    xml_file: str
    frame_skip: int
    default_camera_config: dict[str, float | int | str | None]
    forward_reward_weight: float
    ctrl_cost_weight: float
    healthy_reward: float
    terminate_when_unhealthy: bool
    healthy_state_range: tuple[float, float]
    healthy_z_range: tuple[float, float]
    healthy_angle_range: tuple[float, float]
    reset_noise_scale: float
    exclude_current_positions_from_observation: bool
    camera_id: int | None
    camera_name: str | None
    max_geom: int
    width: int
    height: int
    render_mode: str | None


class Locomotion_2d_MJXEnv(MJXEnv):
    """Base environment class for 2d locomotion environments such as HalfCheetah, Hopper & Walker2d."""

    def __init__(
        self,
        params: Locomotion2dMJXEnvParams = None,  # NOTE not API compliant (yet?)
    ):
        """Sets the `obveration.shape`."""
        if params is None:
            params = self.get_default_params()

        MJXEnv.__init__(self, params=params)

        self.observation_structure = {
            "skipped_qpos": 1 * params["exclude_current_positions_from_observation"],
            "qpos": self.mjx_model.nq
            - 1 * params["exclude_current_positions_from_observation"],
            "qvel": self.mjx_model.nv,
        }

        obs_size = self.observation_structure["qpos"]
        obs_size += self.observation_structure["qvel"]

        self.observation_space = gymnasium.spaces.Box(  # TODO use jnp when and if `Box` supports jax natively
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def observation(
        self,
        state: mjx.Data,
        rng: jax.Array,
        params: Locomotion2dMJXEnvParams,
    ) -> jnp.ndarray:
        """Observes the `qpos` (posional elements) and `qvel` (velocity elements) of the robot."""
        mjx_data = state

        position = mjx_data.qpos.flatten()
        velocity = mjx_data.qvel.flatten()

        if params["exclude_current_positions_from_observation"]:
            position = position[1:]

        observation = jnp.concatenate((position, velocity))
        return observation

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: Locomotion2dMJXEnvParams,
    ) -> tuple[jnp.ndarray, dict]:
        """Reward = foward_reward + healty_reward - control_cost."""
        mjx_data_old = state
        mjx_data_new = next_state

        x_position_before = mjx_data_old.qpos[0]
        x_position_after = mjx_data_new.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt(params)

        forward_reward = params["forward_reward_weight"] * x_velocity
        healthy_reward = params["healthy_reward"] * self._gen_is_healty(
            mjx_data_new, params
        )
        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = params["ctrl_cost_weight"] * jnp.sum(jnp.square(action))

        reward = rewards - costs
        reward_info = {
            "reward_survive": healthy_reward,  # TODO? make optional
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_velocity": x_velocity,
        }

        return reward, reward_info

    def terminal(
        self,
        state: mjx.Data,
        rng: jax.Array,
        params: Locomotion2dMJXEnvParams,
    ) -> bool:
        """Terminates if unhealthy."""
        return jnp.logical_and(
            jnp.logical_not(self._gen_is_healty(state, params)),
            params["terminate_when_unhealthy"],
        )

    def state_info(
        self, state: mjx.Data, params: Locomotion2dMJXEnvParams
    ) -> dict[str, float]:
        """Includes state information exclueded from `observation()`."""
        mjx_data = state

        info = {
            "x_position": mjx_data.qpos[0],
            "z_distance_from_origin": mjx_data.qpos[1] - self.mjx_model.qpos0[1],
        }
        return info

    def _gen_is_healty(self, state: mjx.Data, params: Locomotion2dMJXEnvParams):
        """Checks if the robot is a healthy potision."""
        mjx_data = state

        z, angle = mjx_data.qpos[1:3]
        physics_state = jnp.concatenate(
            (mjx_data.qpos[2:], mjx_data.qvel, mjx_data.act)
        )

        min_state, max_state = params["healthy_state_range"]
        min_z, max_z = params["healthy_z_range"]
        min_angle, max_angle = params["healthy_angle_range"]

        healthy_state = jnp.all(
            jnp.logical_and(min_state < physics_state, physics_state < max_state)
        )
        healthy_z = jnp.logical_and(min_z < z, z < max_z)
        healthy_angle = jnp.logical_and(min_angle < angle, angle < max_angle)

        # NOTE there is probably a clearer way to write this
        is_healthy = jnp.logical_and(
            jnp.logical_and(healthy_state, healthy_z), healthy_angle
        )

        return is_healthy


# The following could maybe be implemented as **kwargs in register()
class HalfCheetahMJXEnv(Locomotion_2d_MJXEnv):
    """Class for HalfCheetah."""

    def _gen_init_physics_state(
        self, rng, params: Locomotion2dMJXEnvParams
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sets `qpos` (positional elements) from a CUD and `qvel` (velocity elements) from a gaussian."""
        noise_low = -params["reset_noise_scale"]
        noise_high = params["reset_noise_scale"]

        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            key=rng, minval=noise_low, maxval=noise_high, shape=(self.mjx_model.nq,)
        )
        qvel = params["reset_noise_scale"] * jax.random.normal(
            key=rng, shape=(self.mjx_model.nv,)
        )
        act = jnp.empty(self.mjx_model.na)

        return qpos, qvel, act

    def get_default_params(self, **kwargs) -> Locomotion2dMJXEnvParams:
        """Get the default parameter for the HalfCheetah environment."""
        default = {
            "xml_file": "half_cheetah.xml",
            "frame_skip": 5,
            "default_camera_config": HALFCHEETAH_DEFAULT_CAMERA_CONFIG,
            "forward_reward_weight": 1.0,
            "ctrl_cost_weight": 0.1,
            "healthy_reward": 0,
            "terminate_when_unhealthy": True,
            "healthy_state_range": (-jnp.inf, jnp.inf),
            "healthy_z_range": (-jnp.inf, jnp.inf),
            "healthy_angle_range": (-jnp.inf, jnp.inf),
            "reset_noise_scale": 0.1,
            "exclude_current_positions_from_observation": True,
        }
        return {**super().get_default_params(), **default, **kwargs}


class HopperMJXEnv(Locomotion_2d_MJXEnv):
    # NOTE: MJX does not yet support condim=1 and therefore this class can not be instantiated
    """Class for Hopper."""

    def _gen_init_physics_state(
        self, rng, params: Locomotion2dMJXEnvParams
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sets `qpos` (positional elements) and `qvel` (velocity elements) form a CUD."""
        noise_low = -params["reset_noise_scale"]
        noise_high = params["reset_noise_scale"]

        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            key=rng, minval=noise_low, maxval=noise_high, shape=(self.mjx_model.nq,)
        )
        qvel = jax.random.uniform(
            key=rng, minval=noise_low, maxval=noise_high, shape=(self.mjx_model.nv,)
        )
        act = jnp.empty(self.mjx_model.na)

        return qpos, qvel, act

    def get_default_params(self, **kwargs) -> Locomotion2dMJXEnvParams:
        """Get the default parameter for the Hopper environment."""
        default = {
            "xml_file": "hopper.xml",
            "frame_skip": 4,
            "default_camera_config": HOPPER_DEFAULT_CAMERA_CONFIG,
            "forward_reward_weight": 1.0,
            "ctrl_cost_weight": 1e-3,
            "healthy_reward": 1.0,
            "terminate_when_unhealthy": True,
            "healthy_state_range": (-100.0, 100.0),
            "healthy_z_range": (0.7, jnp.inf),
            "healthy_angle_range": (-0.2, 0.2),
            "reset_noise_scale": 5e-3,
            "exclude_current_positions_from_observation": True,
        }
        return {**super().get_default_params(), **default, **kwargs}


class Walker2dMJXEnv(Locomotion_2d_MJXEnv):
    """Class for Walker2d."""

    def _gen_init_physics_state(
        self, rng, params: Locomotion2dMJXEnvParams
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sets `qpos` (positional elements) and `qvel` (velocity elements) form a CUD."""
        noise_low = -params["reset_noise_scale"]
        noise_high = params["reset_noise_scale"]

        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            key=rng, minval=noise_low, maxval=noise_high, shape=(self.mjx_model.nq,)
        )
        qvel = jax.random.uniform(
            key=rng, minval=noise_low, maxval=noise_high, shape=(self.mjx_model.nv,)
        )
        act = jnp.empty(self.mjx_model.na)

        return qpos, qvel, act

    def get_default_params(self, **kwargs) -> Locomotion2dMJXEnvParams:
        """Get the default parameter for the Walker2d environment."""
        default = {
            "xml_file": "walker2d_v5.xml",
            "frame_skip": 4,
            "default_camera_config": WALKER2D_DEFAULT_CAMERA_CONFIG,
            "forward_reward_weight": 1.0,
            "ctrl_cost_weight": 1e-3,
            "healthy_reward": 1.0,
            "terminate_when_unhealthy": True,
            "healthy_state_range": (-jnp.inf, jnp.inf),
            "healthy_z_range": (0.8, 2.0),
            "healthy_angle_range": (-1.0, 1.0),
            "reset_noise_scale": 5e-3,
            "exclude_current_positions_from_observation": True,
        }
        return {**super().get_default_params(), **default, **kwargs}


class HalfCheetahJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based HalfCheetah environment using the MJX implementation as base."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs: any):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        default_params = HalfCheetahMJXEnv().get_default_params()
        params = {**default_params, **kwargs}

        env = HalfCheetahMJXEnv(params=params)
        env.transform(jax.jit)

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class HopperJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based Hopper environment using the MJX implementation as base."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs: any):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        default_params = HopperMJXEnv().get_default_params()
        params = {**default_params, **kwargs}

        env = HopperMJXEnv(params=params)
        env.transform(jax.jit)

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class Walker2dJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based Walker2d environment using the MJX implementation as base."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs: any):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        default_params = Walker2dMJXEnv().get_default_params()
        params = {**default_params, **kwargs}

        env = Walker2dMJXEnv(params=params)
        env.transform(jax.jit)

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )
