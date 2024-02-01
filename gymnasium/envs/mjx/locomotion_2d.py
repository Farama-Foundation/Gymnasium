import gymnasium

try:
    import jax
    from jax import numpy as jnp
    from mujoco import mjx
except ImportError as e:
    MJX_IMPORT_ERROR = e
else:
    MJX_IMPORT_ERROR = None

import numpy as np
from typing import Tuple

from gymnasium.envs.mujoco import MujocoRenderer
from gymnasium.envs.mjx.mjx_env import MJXEnv
from gymnasium.envs.mujoco.half_cheetah_v5 import DEFAULT_CAMERA_CONFIG

from typing import Dict


#class Locomotion_2d_Env(MJXEnv, gymnasium.utils.EzPickle):
class Locomotion_2d_Env(MJXEnv):
    """Base environment class for 2d locomotion environments such as HalfCheetah, Hopper & Walker2d."""
    def __init__(
        self,
        xml_file: str,
        frame_skip: int,
        params: Dict[str, any],  # NOTE not API compliant
        #forward_reward_weight: float,
        #ctrl_cost_weight: float,
        #healthy_reward: float,
        #terminate_when_unhealthy: bool,
        #healthy_state_range: Tuple[float, float],
        #healthy_z_range: Tuple[float, float],
        #healthy_angle_range: Tuple[float, float],
        #reset_noise_scale: float,
        #exclude_current_positions_from_observation: bool,
        #**kwargs,
    ):
        """
        gymnasium.utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        """

        MJXEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=frame_skip,
            #**kwargs,
        )

        obs_size = (
            self.mjx_model.nq
            + self.mjx_model.nv
            - params["exclude_current_positions_from_observation"]
        )

        self.observation_space = gymnasium.spaces.Box(  # TODO use jnp when and if `Box` supports jax natively
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.observation_structure = {
            "skipped_qpos": 1 * params["exclude_current_positions_from_observation"],
            "qpos": self.mjx_model.nq - 1 * params["exclude_current_positions_from_observation"],
            "qvel": self.mjx_model.nv,
        }

    def _gen_init_physics_state(self, rng, params: Dict[str, any]) -> "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]":
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

    def observation(self, state: mjx.Data, rng: jax.random.PRNGKey, params: Dict[str, any]) -> jnp.ndarray:
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
        params: Dict[str, any],
    ) -> "tuple[jnp.ndarray, dict]":
        mjx_data_old = state
        mjx_data_new = next_state

        x_position_before = mjx_data_old.qpos[0]
        x_position_after = mjx_data_new.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        forward_reward = params["forward_reward_weight"] * x_velocity
        healthy_reward = params["healthy_reward"] * float(self._gen_is_healty(mjx_data_new, params))
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

    def terminal(self, state: mjx.Data, rng: jax.random.PRNGKey, params: Dict[str, any]) -> bool:
        return (not self._gen_is_healty(state, params)) and params["terminate_when_unhealthy"]

    def state_info(self, state: mjx.Data, params: Dict[str, any]) -> dict[str, float]:
        mjx_data = state
        x_position_after = mjx_data.qpos[0]
        info = {
            "x_position": x_position_after,
        }
        return info

    def render_init(
        self, default_camera_config: "dict[str, float]" = DEFAULT_CAMERA_CONFIG, **kwargs
    ) -> MujocoRenderer:
        return super().render_init(
            default_camera_config=default_camera_config, **kwargs
        )

    def _gen_is_healty(self, state: mjx.Data, params: Dict[str, any]):
        mjx_data = state

        z, angle = mjx_data.qpos[1:3]
        physics_state = jnp.concatenate((mjx_data.qpos[2:], mjx_data.qvel, mjx_data.act))

        min_state, max_state = params["healthy_state_range"]
        min_z, max_z = params["healthy_z_range"]
        min_angle, max_angle = params["healthy_angle_range"]

        healthy_state = jnp.all(jnp.logical_and(min_state < physics_state, physics_state < max_state))
        healthy_z = jnp.logical_and(min_z < z, z < max_z)
        healthy_angle = jnp.logical_and(min_angle < angle, angle < max_angle)

        return False
        #is_healthy = all((healthy_state, healthy_z, healthy_angle))
        #is_healthy = jnp.all(jnp.concatenate((healthy_state, healthy_z, healthy_angle)))
        #is_healthy = bool(healthy_state) and bool(healthy_z) and bool(healthy_angle)
        is_healthy = healthy_state
        return is_healthy


# The following could be implemented as **kwargs in register()
# TODO fix camera configs
class HalfCheetahMJXEnv(Locomotion_2d_Env):
    def __init__(
        self,
        params: Dict[str, any],
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        #forward_reward_weight: float = 1.0,
        #ctrl_cost_weight: float = 0.1,
        #healthy_reward: float = 0,
        #terminate_when_unhealthy: bool = True,
        #healthy_state_range: Tuple[float, float] = (-jnp.inf, jnp.inf),
        #healthy_z_range: Tuple[float, float] = (-jnp.inf, jnp.inf),
        #healthy_angle_range: Tuple[float, float] = (-jnp.inf, jnp.inf),
        #reset_noise_scale: float = 0.1,
        #exclude_current_positions_from_observation: bool = True,
        #**kwargs,
    ):
        super().__init__(
            params=params,
            xml_file=xml_file,
            frame_skip=frame_skip,
            #forward_reward_weight=forward_reward_weight,
            #ctrl_cost_weight=ctrl_cost_weight,
            #healthy_reward=healthy_reward,
            #terminate_when_unhealthy=terminate_when_unhealthy,
            #healthy_state_range=healthy_state_range,
            #healthy_z_range=healthy_z_range,
            #healthy_angle_range=healthy_angle_range,
            #reset_noise_scale=reset_noise_scale,
            #exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            #**kwargs,
        )

    def get_default_params(**kwargs) -> dict[str, any]:
        default = {
            "xml_file": "half_cheetah.xml",
            "frame_skip": 5,
            "forward_reward_weight": 1.0,
            "ctrl_cost_weight": 0.1,
            "healthy_reward": 0,
            "terminate_when_unhealthy": True,
            "healthy_state_range": (-jnp.inf, jnp.inf),
            "healthy_z_range": (-jnp.inf, jnp.inf),
            "healthy_angle_range": (-jnp.inf, jnp.inf),
            "reset_noise_scale" : 0.1,
            "exclude_current_positions_from_observation": True,
        }
        return {**default, **kwargs}


class HopperMJXEnv(Locomotion_2d_Env):
    def __init__(
        self,
        xml_file: str = "hopper.xml",
        frame_skip: int = 4,
        #forward_reward_weight: float = 1.0,
        #ctrl_cost_weight: float = 1e-3,
        #healthy_reward: float = 1.0,
        #terminate_when_unhealthy: bool = True,
        #healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        #healthy_z_range: Tuple[float, float] = (0.7, jnp.inf),
        #healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        #reset_noise_scale: float = 5e-3,
        #exclude_current_positions_from_observation: bool = True,
        #**kwargs,
    ):
        super().__init__(
            xml_file=xml_file,
            frame_skip=frame_skip,
            #forward_reward_weight=forward_reward_weight,
            #ctrl_cost_weight=ctrl_cost_weight,
            #healthy_reward=healthy_reward,
            #terminate_when_unhealthy=terminate_when_unhealthy,
            #healthy_state_range=healthy_state_range,
            #healthy_z_range=healthy_z_range,
            #healthy_angle_range=healthy_angle_range,
            #reset_noise_scale=reset_noise_scale,
            #exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            #**kwargs,
        )


class Walker2dMJXEnv(Locomotion_2d_Env):
    def __init__(
        self,
        xml_file: str = "walker2d_v5.xml",
        frame_skip: int = 4,
        #forward_reward_weight: float = 1.0,
        #ctrl_cost_weight: float = 1e-3,
        #healthy_reward: float = 1.0,
        #terminate_when_unhealthy: bool = True,
        #healthy_state_range: Tuple[float, float] = (-jnp.inf, jnp.inf),
        #healthy_z_range: Tuple[float, float] = (0.8, 2.0),
        #healthy_angle_range: Tuple[float, float] = (-1.0, 1.0),
        #reset_noise_scale: float = 5e-3,
        #exclude_current_positions_from_observation: bool = True,
        #**kwargs,
    ):
        super().__init__(
            xml_file=xml_file,
            frame_skip=frame_skip,
            #forward_reward_weight=forward_reward_weight,
            #ctrl_cost_weight=ctrl_cost_weight,
            #healthy_reward=healthy_reward,
            #terminate_when_unhealthy=terminate_when_unhealthy,
            #healthy_state_range=healthy_state_range,
            #healthy_z_range=healthy_z_range,
            #healthy_angle_range=healthy_angle_range,
            #reset_noise_scale=reset_noise_scale,
            #exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            #**kwargs,
        )
