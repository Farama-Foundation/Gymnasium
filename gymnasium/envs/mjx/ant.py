"""Contains the class for the `Ant` environment."""
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

from gymnasium.envs.mujoco.ant_v5 import DEFAULT_CAMERA_CONFIG


class Ant_MJXEnv(MJXEnv):
    # NOTE: MJX does not yet support cfrc_ext and therefore this class can not be instantiated
    """Class for Ant."""

    def __init__(
        self,
        params: Dict[str, any],
    ):
        """Sets the `obveration_space`."""
        MJXEnv.__init__(self, params=params)

        obs_size = (
            self.mjx_model.nq
            + self.mjx_model.nv
            - 2 * params["exclude_current_positions_from_observation"]
            + (self.mjx_model.nbody - 1) * 6 * params["include_cfrc_ext_in_observation"]
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * params["exclude_current_positions_from_observation"],
            "qpos": self.mjx_model.nq
            - 2 * params["exclude_current_positions_from_observation"],
            "qvel": self.mjx_model.nv,
            "cfrc_ext": (self.mjx_model.nbody - 1)
            * 6
            * params["include_cfrc_ext_in_observation"],
        }

    def _gen_init_physics_state(
        self, rng, params: Dict[str, any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        self, state: mjx.Data, rng: jax.random.PRNGKey, params: Dict[str, any]
    ) -> jnp.ndarray:
        """Observes the `qpos` (posional elements) and `qvel` (velocity elements) and `cfrc_ext` (external contact forces) of the robot."""
        mjx_data = state

        position = mjx_data.qpos.flatten()
        velocity = mjx_data.qvel.flatten()

        if params["exclude_current_positions_from_observation"]:
            position = position[2:]

        if self._include_cfrc_ext_in_observation:
            contact_force = self._get_contact_forces(mjx_data, params)
            observation = jnp.concatenate((position, velocity, contact_force))
        else:
            observation = jnp.concatenate((position, velocity))

        return observation

    def _get_contact_forces(self, mjx_data: mjx.Data, params: Dict[str, any]):
        """Get External Contact Forces (`cfrc_ext`) clipped by `contact_force_range`."""
        raw_contact_forces = mjx_data.cfrc_ext
        min_value, max_value = params["contact_force_range"]
        contact_forces = jnp.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: Dict[str, any],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Reward = forward_reward + healthy_reward - ctrl_cost - contact cost."""
        mjx_data_old = state
        mjx_data_new = next_state

        xy_position_before = mjx_data_old.xpos[params["main_body"], :2]
        xy_position_after = mjx_data_new.xpos[params["main_body"], :2]

        xy_velocity = (xy_position_after - xy_position_before) / self.dt(params)
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity * params["forward_reward_weight"]
        healthy_reward = (
            self._gen_is_healthy(mjx_data_new, params) * params["healthy_reward"]
        )
        rewards = forward_reward + healthy_reward

        ctrl_cost = params["ctrl_cost_weight"] * jnp.sum(jnp.square(action))
        contact_cost = params["contact_cost_weight"] * jnp.sum(
            jnp.square(self._get_contact_forces(mjx_data_new, params))
        )
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _gen_is_healty(self, state: mjx.Data, params: Dict[str, any]) -> jnp.ndarray:
        """Checks if the robot is in a healthy potision."""
        mjx_data = state

        z = mjx_data.qpos[2]
        min_z, max_z = params["healthy_z_range"]
        is_healthy = (
            jnp.isfinite(
                jnp.concatenate(mjx_data.qpos, mjx_data.qvel.mjx_data.act)
            ).all()
            and min_z <= z <= max_z
        )
        return is_healthy

    def state_info(self, state: mjx.Data, params: Dict[str, any]) -> Dict[str, float]:
        """Includes state information exclueded from `observation()`."""
        mjx_data = state

        info = {
            "x_position": mjx_data.qpos[0],
            "y_position": mjx_data.qpos[1],
            "distance_from_origin": jnp.linalg.norm(mjx_data.qpos[0:2], ord=2),
        }
        return info

    def terminal(
        self, state: mjx.Data, rng: jax.random.PRNGKey, params: Dict[str, any]
    ) -> bool:
        """Terminates if unhealthy."""
        return jnp.logical_and(
            jnp.logical_not(self._gen_is_healty(state, params)),
            params["terminate_when_unhealthy"],
        )

    def get_default_params(**kwargs) -> Dict[str, any]:
        """Get the default parameter for the Ant environment."""
        default = {
            "xml_file": "ant.xml",
            "frame_skip": 5,
            "default_camera_config": DEFAULT_CAMERA_CONFIG,
            "forward_reward_weight": 1,
            "ctrl_cost_weight": 0.5,
            "contact_cost_weight": 5e-4,
            "healthy_reward": 1.0,
            "main_body": 1,
            "terminate_when_unhealthy": True,
            "healthy_z_range": (0.2, 1.0),
            "contact_force_range": (-1.0, 1.0),
            "reset_noise_scale": 0.1,
            "exclude_current_positions_from_observation": True,
            "include_cfrc_ext_in_observation": True,
        }
        return {**MJXEnv.get_default_params(), **default, **kwargs}
