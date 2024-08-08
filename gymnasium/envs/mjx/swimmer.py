"""Contains the class for the `Swimmer` environment."""
import gymnasium


try:
    import jax
    from jax import numpy as jnp
    from mujoco import mjx
except ImportError as e:
    MJX_IMPORT_ERROR = e
else:
    MJX_IMPORT_ERROR = None

from typing import Dict, Tuple

import numpy as np

from gymnasium.envs.mjx.mjx_env import MJXEnv


class Swimmer_MJXEnv(MJXEnv):
    # NOTE: MJX does not yet support condim=1 and therefore this class can not be instantiated
    """Class for Swimmer."""

    def __init__(
        self,
        params: Dict[str, any],
    ):
        """Sets the `obveration_space`."""
        MJXEnv.__init__(self, params=params)

        self.observation_structure = {
            "skipped_qpos": 2 * params["exclude_current_positions_from_observation"],
            "qpos": self.mjx_model.nq
            - 2 * params["exclude_current_positions_from_observation"],
            "qvel": self.mjx_model.nv,
        }

        obs_size = self.observation_structure["qpos"]
        obs_size += self.observation_structure["qvel"]

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def _gen_init_physics_state(
        self, rng, params: Dict[str, any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        self, state: mjx.Data, rng: jax.random.PRNGKey, params: Dict[str, any]
    ) -> jnp.ndarray:
        """Observes the `qpos` (posional elements) and `qvel` (velocity elements) of the robot."""
        mjx_data = state

        position = mjx_data.qpos.flatten()
        velocity = mjx_data.qvel.flatten()

        if params["exclude_current_positions_from_observation"]:
            position = position[2:]

        observation = jnp.concatenate((position, velocity))
        return observation

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: Dict[str, any],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Reward = reward_dist + reward_ctrl."""
        mjx_data_old = state
        mjx_data_new = next_state

        x_position_before = mjx_data_old.qpos[0]
        x_position_after = mjx_data_new.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt(params)

        forward_reward = params["forward_reward_weight"] * x_velocity
        ctrl_cost = params["ctrl_cost_weight"] * jnp.sum(jnp.square(action))

        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def state_info(self, state: mjx.Data, params: Dict[str, any]) -> Dict[str, float]:
        """Includes state information exclueded from `observation()`."""
        mjx_data = state

        info = {
            "x_position": mjx_data.qpos[0],
            "y_position": mjx_data.qpos[1],
            "distance_from_origin": jnp.linalg.norm(mjx_data.qpos[0:2], ord=2),
        }
        return info

    def get_default_params(**kwargs) -> Dict[str, any]:
        """Get the default parameter for the Swimmer environment."""
        default = {
            "xml_file": "swimmer.xml",
            "frame_skip": 4,
            "default_camera_config": {},
            "forward_reward_weight": 1.0,
            "ctrl_cost_weight": 1e-4,
            "reset_noise_scale": 0.1,
            "exclude_current_positions_from_observation": True,
        }
        return {**MJXEnv.get_default_params(), **default, **kwargs}
