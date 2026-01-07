"""Contains the classes for the humaanoid environments environments, `Humanoid` and `HumanoidStandup`."""

import gymnasium


try:
    import jax
    from jax import numpy as jnp
    from mujoco import mjx
except ImportError as e:
    MJX_IMPORT_ERROR = e
else:
    MJX_IMPORT_ERROR = None

from typing import TypedDict, Union

import numpy as np

from gymnasium.envs.mjx.mjx_env import MJXEnv
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco.humanoid_v5 import (
    DEFAULT_CAMERA_CONFIG as HUMANOID_DEFAULT_CAMERA_CONFIG,
)
from gymnasium.envs.mujoco.humanoidstandup_v5 import (
    DEFAULT_CAMERA_CONFIG as HUMANOIDSTANDUP_DEFAULT_CAMERA_CONFIG,
)


class HumanoidMJXEnvParams(TypedDict):
    """Parameters for the Humanoid environment."""

    xml_file: str
    frame_skip: int
    default_camera_config: dict[str, float | int | str | None]
    forward_reward_weight: float
    ctrl_cost_weight: float
    contact_cost_weight: float
    contact_cost_range: tuple[float, float]
    healthy_reward: float
    terminate_when_unhealthy: bool
    healthy_z_range: tuple[float, float]
    reset_noise_scale: float
    exclude_current_positions_from_observation: bool
    include_cinert_in_observation: bool
    include_cvel_in_observation: bool
    include_qfrc_actuator_in_observation: bool
    include_cfrc_ext_in_observation: bool
    camera_id: int | None
    camera_name: str | None
    max_geom: int
    width: int
    height: int
    render_mode: str | None


class HumanoidStandupMJXEnvParams(TypedDict):
    """Parameters for the HumanoidStandup environment."""

    xml_file: str
    frame_skip: int
    default_camera_config: dict[str, float | int | str | None]
    uph_cost_weight: float
    ctrl_cost_weight: float
    impact_cost_weight: float
    impact_cost_range: tuple[float, float]
    reset_noise_scale: float
    exclude_current_positions_from_observation: bool
    include_cinert_in_observation: bool
    include_cvel_in_observation: bool
    include_qfrc_actuator_in_observation: bool
    include_cfrc_ext_in_observation: bool
    camera_id: int | None
    camera_name: str | None
    max_geom: int
    width: int
    height: int
    render_mode: str | None


BaseHumanoidMJXEnvParams = Union[HumanoidMJXEnvParams, HumanoidStandupMJXEnvParams]


class BaseHumanoid_MJXEnv(MJXEnv):
    # NOTE: MJX does not yet support many features therefore this class can not be instantiated
    """Base environment class for humanoid environments such as Humanoid, & HumanoidStandup."""

    def __init__(
        self,
        params: BaseHumanoidMJXEnvParams = None,
    ):
        """Sets the `obveration_space`."""
        if params is None:
            params = self.get_default_params()

        MJXEnv.__init__(self, params=params)

        self.observation_structure = {
            "skipped_qpos": 2 * params["exclude_current_positions_from_observation"],
            "qpos": self.mjx_model.nq
            - 2 * params["exclude_current_positions_from_observation"],
            "qvel": self.mjx_model.nv,
            "cinert": (self.mjx_model.nbody - 1)
            * 10
            * params["include_cinert_in_observation"],
            "cvel": (self.mjx_model.nbody - 1)
            * 6
            * params["include_cvel_in_observation"],
            "qfrc_actuator": (self.mjx_model.nv - 6)
            * params["include_qfrc_actuator_in_observation"],
            "cfrc_ext": (self.mjx_model.nbody - 1)
            * 6
            * params["include_cfrc_ext_in_observation"],
            "ten_lenght": 0,
            "ten_velocity": 0,
        }

        obs_size = self.observation_structure["qpos"]
        obs_size += self.observation_structure["qvel"]
        obs_size += self.observation_structure["cinert"]
        obs_size += self.observation_structure["cvel"]
        obs_size += self.observation_structure["qfrc_actuator"]
        obs_size += self.observation_structure["cfrc_ext"]

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def observation(
        self,
        state: mjx.Data,
        rng: jax.Array,
        params: BaseHumanoidMJXEnvParams,
    ) -> jnp.ndarray:
        """Observes the `qpos` (posional elements) and `qvel` (velocity elements) and `cfrc_ext` (external contact forces) of the robot."""
        mjx_data = state

        position = mjx_data.qpos.flatten()
        velocity = mjx_data.qvel.flatten()

        if params["exclude_current_positions_from_observation"]:
            position = position[2:]

        if params["include_cinert_in_observation"] is True:
            com_inertia = mjx_data.cinert[1:].flatten()
        else:
            com_inertia = jnp.array([])
        if params["include_cvel_in_observation"] is True:
            com_velocity = mjx_data.cvel[1:].flatten()
        else:
            com_velocity = jnp.array([])

        if params["include_qfrc_actuator_in_observation"] is True:
            actuator_forces = mjx_data.qfrc_actuator[6:].flatten()
        else:
            actuator_forces = jnp.array([])
        if params["include_cfrc_ext_in_observation"] is True:
            external_contact_forces = mjx_data._impl.cfrc_ext[1:].flatten()
        else:
            external_contact_forces = jnp.array([])

        observation = jnp.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )
        return observation

    def state_info(
        self, state: mjx.Data, params: BaseHumanoidMJXEnvParams
    ) -> dict[str, float]:
        """Includes state information exclueded from `observation()`."""
        mjx_data = state

        info = {
            "x_position": mjx_data.qpos[0],
            "y_position": mjx_data.qpos[1],
            "tendon_lenght": mjx_data.ten_length,
            "tendon_velocity": mjx_data.ten_velocity,
            "distance_from_origin": jnp.linalg.norm(mjx_data.qpos[0:2], ord=2),
        }
        return info

    def _gen_init_physics_state(
        self, rng, params: BaseHumanoidMJXEnvParams
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


class HumanoidMJXEnv(BaseHumanoid_MJXEnv):
    """Class for Humanoid."""

    def mass_center(self, mjx_data):
        """Calculates the xpos based center of mass."""
        mass = np.expand_dims(self.mjx_model.body_mass, axis=1)
        xpos = mjx_data.xipos
        return (jnp.sum(mass * xpos, axis=0) / jnp.sum(mass))[0:2]

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: HumanoidMJXEnvParams,
    ) -> tuple[jnp.ndarray, dict]:
        """Reward = forward_reward + healthy_reward - ctrl_cost - contact_cost."""
        mjx_data_old = state
        mjx_data_new = next_state

        xy_position_before = self.mass_center(mjx_data_old)
        xy_position_after = self.mass_center(mjx_data_new)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt(params)
        x_velocity, y_velocity = xy_velocity

        forward_reward = params["forward_reward_weight"] * x_velocity
        healthy_reward = params["healthy_reward"] * self._gen_is_healty(
            mjx_data_new, params
        )
        rewards = forward_reward + healthy_reward

        ctrl_cost = params["ctrl_cost_weight"] * jnp.sum(jnp.square(action))
        contact_cost = self._get_conctact_cost(mjx_data_new, params)
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }

        return reward, reward_info

    def _get_conctact_cost(self, mjx_data: mjx.Data, params: HumanoidMJXEnvParams):
        contact_forces = mjx_data._impl.cfrc_ext
        contact_cost = params["contact_cost_weight"] * jnp.sum(
            jnp.square(contact_forces)
        )
        min_cost, max_cost = params["contact_cost_range"]
        contact_cost = jnp.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    def _gen_is_healty(self, state: mjx.Data, params: HumanoidMJXEnvParams):
        """Checks if the robot is in a healthy potision."""
        mjx_data = state

        min_z, max_z = params["healthy_z_range"]
        is_healthy = min_z < mjx_data.qpos[2] < max_z

        return is_healthy

    def terminal(
        self, state: mjx.Data, rng: jax.Array, params: HumanoidMJXEnvParams
    ) -> bool:
        """Terminates if unhealthy."""
        return jnp.logical_and(
            jnp.logical_not(self._gen_is_healty(state, params)),
            params["terminate_when_unhealthy"],
        )

    def get_default_params(self, **kwargs) -> HumanoidMJXEnvParams:
        """Get the default parameter for the Humanoid environment."""
        default = {
            "xml_file": "humanoid.xml",
            "frame_skip": 5,
            "default_camera_config": HUMANOID_DEFAULT_CAMERA_CONFIG,
            "forward_reward_weight": 1.25,
            "ctrl_cost_weight": 0.1,
            "contact_cost_weight": 5e-7,
            "contact_cost_range": (-np.inf, 10.0),
            "healthy_reward": 5.0,
            "terminate_when_unhealthy": True,
            "healthy_z_range": (1.0, 2.0),
            "reset_noise_scale": 1e-2,
            "exclude_current_positions_from_observation": True,
            "include_cinert_in_observation": True,
            "include_cvel_in_observation": True,
            "include_qfrc_actuator_in_observation": True,
            "include_cfrc_ext_in_observation": True,
        }
        return {**super().get_default_params(), **default, **kwargs}


class HumanoidStandupMJXEnv(BaseHumanoid_MJXEnv):
    """Class for HumanoidStandup."""

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: HumanoidStandupMJXEnvParams,
    ) -> tuple[jnp.ndarray, dict]:
        """Reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1."""
        mjx_data_new = next_state

        pos_after = mjx_data_new.qpos[2]

        uph_cost = (pos_after - 0) / self.mjx_model.opt.timestep

        quad_ctrl_cost = params["ctrl_cost_weight"] * jnp.square(action).sum()

        quad_impact_cost = (
            params["impact_cost_weight"] * jnp.square(mjx_data_new._impl.cfrc_ext).sum()
        )
        min_impact_cost, max_impact_cost = params["impact_cost_range"]
        quad_impact_cost = jnp.clip(quad_impact_cost, min_impact_cost, max_impact_cost)

        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        reward_info = {
            "reward_linup": uph_cost,
            "reward_quadctrl": -quad_ctrl_cost,
            "reward_impact": -quad_impact_cost,
        }

        return reward, reward_info

    def get_default_params(self, **kwargs) -> HumanoidStandupMJXEnvParams:
        """Get the default parameter for the Humanoid environment."""
        default = {
            "xml_file": "humanoidstandup.xml",
            "frame_skip": 5,
            "default_camera_config": HUMANOIDSTANDUP_DEFAULT_CAMERA_CONFIG,
            "uph_cost_weight": 1,
            "ctrl_cost_weight": 0.1,
            "impact_cost_weight": 0.5e-6,
            "impact_cost_range": (-np.inf, 10.0),
            "reset_noise_scale": 1e-2,
            "exclude_current_positions_from_observation": True,
            "include_cinert_in_observation": True,
            "include_cvel_in_observation": True,
            "include_qfrc_actuator_in_observation": True,
            "include_cfrc_ext_in_observation": True,
        }
        return {**super().get_default_params(), **default, **kwargs}


class HumanoidJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based Humanoid environment using the MJX implementation as base."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs: any):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        temp_env = HumanoidMJXEnv()
        params = temp_env.get_default_params(**kwargs)

        env = HumanoidMJXEnv(params=params)
        env.transform(jax.jit)

        metadata = dict(env.metadata)
        metadata["jax"] = True

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=metadata,
            render_mode=render_mode,
        )


class HumanoidStandupJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based HumanoidStandup environment using the MJX implementation as base."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs: any):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        temp_env = HumanoidStandupMJXEnv()
        params = temp_env.get_default_params(**kwargs)

        env = HumanoidStandupMJXEnv(params=params)
        env.transform(jax.jit)

        metadata = dict(env.metadata)
        metadata["jax"] = True

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=metadata,
            render_mode=render_mode,
        )
