"""Contains the classes for the Inverted Pendulum environments, `InvertedPendulum`, `InvertedDoublePendulum`."""

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
from gymnasium.envs.mujoco.inverted_double_pendulum_v5 import (
    DEFAULT_CAMERA_CONFIG as INVERTED_DOUBLE_PENDULUM_DEFAULT_CAMERA_CONFIG,
)
from gymnasium.envs.mujoco.inverted_pendulum_v5 import (
    DEFAULT_CAMERA_CONFIG as INVERTED_PENDULUM_DEFAULT_CAMERA_CONFIG,
)


class InvertedDoublePendulumMJXEnvParams(TypedDict):
    """Parameters for the InvertedDoublePendulum environment."""

    xml_file: str
    frame_skip: int
    default_camera_config: dict[str, float | int | str | None]
    healthy_reward: float
    reset_noise_scale: float
    camera_id: int | None
    camera_name: str | None
    max_geom: int
    width: int
    height: int
    render_mode: str | None


class InvertedPendulumMJXEnvParams(TypedDict):
    """Parameters for the InvertedPendulum environment."""

    xml_file: str
    frame_skip: int
    default_camera_config: dict[str, float | int | str | None]
    reset_noise_scale: float
    camera_id: int | None
    camera_name: str | None
    max_geom: int
    width: int
    height: int
    render_mode: str | None


class InvertedDoublePendulumMJXEnv(MJXEnv):
    """Class for InvertedDoublePendulum."""

    def __init__(
        self,
        params: InvertedDoublePendulumMJXEnvParams = None,  # NOTE not API compliant (yet?)
    ):
        """Sets the `obveration_space.shape`."""
        if params is None:
            params = self.get_default_params()

        MJXEnv.__init__(self, params=params)

        # TODO use jnp when and if `Box` supports jax natively
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

    def _gen_init_physics_state(
        self, rng, params: InvertedDoublePendulumMJXEnvParams
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

    def observation(
        self,
        state: mjx.Data,
        rng: jax.Array,
        params: InvertedDoublePendulumMJXEnvParams,
    ) -> jnp.ndarray:
        """Observes the `qpos` (posional elements) and `qvel` (velocity elements) of the robot."""
        mjx_data = state

        velocity = mjx_data.qvel.flatten()

        observation = jnp.concatenate(
            (
                mjx_data.qpos.flatten()[:1],  # `cart` x-position
                jnp.sin(mjx_data.qpos[1:]),
                jnp.cos(mjx_data.qpos[1:]),
                jnp.clip(velocity, -10, 10),
                jnp.clip(mjx_data.qfrc_constraint, -10, 10)[:1],
            )
        )
        return observation

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: InvertedDoublePendulumMJXEnvParams,
    ) -> tuple[jnp.ndarray, dict]:
        """Reward = alive_bonus - dist_penalty - vel_penalty."""
        mjx_data_new = next_state

        v = mjx_data_new.qvel[1:3]
        x, _, y = mjx_data_new.site_xpos[0]

        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        vel_penalty = jnp.array([1e-3, 5e-3]).T * jnp.square(v)
        alive_bonus = params["healthy_reward"] * self._gen_is_healty(mjx_data_new)

        reward = alive_bonus - dist_penalty - vel_penalty

        reward_info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
        }

        return reward, reward_info

    def _gen_is_healty(self, state: mjx.Data):
        """Checks if the pendulum is upright."""
        mjx_data = state

        y = mjx_data.site_xpos[0][2]

        return jnp.array(y > 1)

    def terminal(
        self,
        state: mjx.Data,
        rng: jax.Array,
        params: InvertedDoublePendulumMJXEnvParams,
    ) -> bool:
        """Terminates if unhealty."""
        return jnp.logical_not(self._gen_is_healty(state))

    def get_default_params(self, **kwargs) -> InvertedDoublePendulumMJXEnvParams:
        """Get the parameters for the InvertedDoublePendulum environment."""
        default = {
            "xml_file": "inverted_double_pendulum.xml",
            "frame_skip": 5,
            "default_camera_config": INVERTED_DOUBLE_PENDULUM_DEFAULT_CAMERA_CONFIG,
            "healthy_reward": 10.0,
            "reset_noise_scale": 0.1,
        }
        return {**super().get_default_params(), **default, **kwargs}


class InvertedPendulumMJXEnv(MJXEnv):
    """Class for InvertedPendulum."""

    def __init__(
        self,
        params: InvertedPendulumMJXEnvParams = None,  # NOTE not API compliant (yet?)
    ):
        """Sets the `obveration_space.shape`."""
        if params is None:
            params = self.get_default_params()

        MJXEnv.__init__(self, params=params)

        self.observation_structure = {
            "qpos": self.mjx_model.nq,
            "qvel": self.mjx_model.nv,
        }

        obs_size = self.observation_structure["qpos"]
        obs_size += self.observation_structure["qvel"]

        # TODO use jnp when and if `Box` supports jax natively
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _gen_init_physics_state(
        self, rng, params: InvertedPendulumMJXEnvParams
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

    def observation(
        self,
        state: mjx.Data,
        rng: jax.Array,
        params: InvertedPendulumMJXEnvParams,
    ) -> jnp.ndarray:
        """Observes the `qpos` (posional elements) and `qvel` (velocity elements) of the robot."""
        mjx_data = state

        position = mjx_data.qpos.flatten()
        velocity = mjx_data.qvel.flatten()

        observation = jnp.concatenate((position, velocity))
        return observation

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: InvertedPendulumMJXEnvParams,
    ) -> tuple[jnp.ndarray, dict]:
        reward = jnp.array(self._gen_is_healty(next_state), dtype=jnp.float32)
        reward_info = {"reward_survive": reward}
        return reward, reward_info

    def _gen_is_healty(self, state: mjx.Data):
        """Checks if the pendulum is upright."""
        mjx_data = state

        angle = mjx_data.qpos[1]

        return jnp.abs(angle) <= 0.2

    def terminal(
        self,
        state: mjx.Data,
        rng: jax.Array,
        params: InvertedPendulumMJXEnvParams,
    ) -> bool:
        """Terminates if unhealty."""
        return jnp.logical_not(self._gen_is_healty(state))

    def get_default_params(self, **kwargs) -> InvertedPendulumMJXEnvParams:
        """Get the parameters for the InvertedPendulum environment."""
        default = {
            "xml_file": "inverted_pendulum.xml",
            "frame_skip": 2,
            "default_camera_config": INVERTED_PENDULUM_DEFAULT_CAMERA_CONFIG,
            "reset_noise_scale": 0.01,
        }

        return {**super().get_default_params(), **default, **kwargs}


class InvertedDoublePendulumJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based InvertedDoublePendulum environment using the MJX implementation as base."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs: any):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        temp_env = InvertedDoublePendulumMJXEnv()
        params = temp_env.get_default_params(**kwargs)

        env = InvertedDoublePendulumMJXEnv(params=params)
        env.transform(jax.jit)

        metadata = dict(env.metadata)
        metadata["jax"] = True

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=metadata,
            render_mode=render_mode,
        )


class InvertedPendulumJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based InvertedPendulum environment using the MJX implementation as base."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs: any):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        temp_env = InvertedPendulumMJXEnv()
        params = temp_env.get_default_params(**kwargs)

        env = InvertedPendulumMJXEnv(params=params)
        env.transform(jax.jit)

        metadata = dict(env.metadata)
        metadata["jax"] = True

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=metadata,
            render_mode=render_mode,
        )
