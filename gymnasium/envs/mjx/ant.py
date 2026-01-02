"""Contains the class for the `Ant` environment."""

import gymnasium
from functools import partial


try:
    import jax
    from jax import numpy as jnp
    from mujoco import mjx
    import flax.struct
except ImportError as e:
    MJX_IMPORT_ERROR = e
else:
    MJX_IMPORT_ERROR = None

import numpy as np

from gymnasium.envs.mjx.mjx_env import MJXEnv
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv, FunctionalJaxVectorEnv
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco.ant_v5 import DEFAULT_CAMERA_CONFIG


@flax.struct.dataclass
class AntMJXEnvParams:
    """Parameters for the Ant environment."""

    # xml_file: str
    xml_file: int  # TODO temporarly change to int to bypass MJX xml loading issue
    frame_skip: int
    default_camera_config: dict[str, float | int | str | None]
    forward_reward_weight: float
    ctrl_cost_weight: float
    contact_cost_weight: float
    healthy_reward: float
    main_body: int
    terminate_when_unhealthy: bool
    healthy_z_range: tuple[float, float]
    contact_force_range: tuple[float, float]
    reset_noise_scale: float
    exclude_current_positions_from_observation: bool
    include_cfrc_ext_in_observation: bool
    camera_id: int | None
    camera_name: str | None
    max_geom: int
    width: int
    height: int
    render_mode: str | None


class Ant_MJXEnv(MJXEnv):
    # NOTE: MJX does not yet support cfrc_ext and therefore this class can not be instantiated
    """Class for Ant."""

    def __init__(
        self,
        params: AntMJXEnvParams = None,
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
            "cfrc_ext": (self.mjx_model.nbody - 1)
            * 6
            * params.include_cfrc_ext_in_observation,
        }

        obs_size = self.observation_structure["qpos"]
        obs_size += self.observation_structure["qvel"]
        obs_size += self.observation_structure["cfrc_ext"]

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def _gen_init_physics_state(
        self, rng, params: AntMJXEnvParams
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sets `qpos` (positional elements) from a CUD and `qvel` (velocity elements) from a gaussian."""
        noise_low = -params.reset_noise_scale
        noise_high = params.reset_noise_scale

        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            key=rng, minval=noise_low, maxval=noise_high, shape=(self.mjx_model.nq,)
        )
        qvel = params.reset_noise_scale * jax.random.normal(
            key=rng, shape=(self.mjx_model.nv,)
        )
        act = jnp.empty(self.mjx_model.na)

        return qpos, qvel, act

    def observation(
        self, state: mjx.Data, rng: jax.Array, params: AntMJXEnvParams
    ) -> jnp.ndarray:
        """Observes the `qpos` (posional elements) and `qvel` (velocity elements) and `cfrc_ext` (external contact forces) of the robot."""
        mjx_data = state

        position = mjx_data.qpos.flatten()
        velocity = mjx_data.qvel.flatten()

        if params.exclude_current_positions_from_observation:
            position = position[2:]

        if params.include_cfrc_ext_in_observation is True:
            external_contact_forces = self._get_contact_forces(mjx_data, params)
        else:
            external_contact_forces = jnp.array([])

        observation = jnp.concatenate((position, velocity, external_contact_forces))

        return observation

    def _get_contact_forces(self, mjx_data: mjx.Data, params: AntMJXEnvParams):
        """Get External Contact Forces (`cfrc_ext`) clipped by `contact_force_range`."""
        raw_contact_forces = mjx_data._impl.cfrc_ext
        min_value, max_value = params.contact_force_range
        contact_forces = jnp.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: AntMJXEnvParams,
    ) -> tuple[jnp.ndarray, dict]:
        """Reward = forward_reward + healthy_reward - ctrl_cost - contact cost."""
        mjx_data_old = state
        mjx_data_new = next_state

        xy_position_before = mjx_data_old.xpos[params.main_body, :2]
        xy_position_after = mjx_data_new.xpos[params.main_body, :2]

        xy_velocity = (xy_position_after - xy_position_before) / self.dt(params)
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity * params.forward_reward_weight
        healthy_reward = (
            self._gen_is_healthy(mjx_data_new, params) * params.healthy_reward
        )
        rewards = forward_reward + healthy_reward

        ctrl_cost = params.ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = params.contact_cost_weight * jnp.sum(
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

    def _gen_is_healthy(self, state: mjx.Data, params: AntMJXEnvParams) -> jnp.ndarray:
        """Checks if the robot is in a healthy potision."""
        mjx_data = state

        z = mjx_data.qpos[2]
        min_z, max_z = params.healthy_z_range
        is_finite = jnp.isfinite(
            jnp.concatenate((mjx_data.qpos, mjx_data.qvel, mjx_data.act))
        ).all()
        z_ok = jnp.logical_and(z >= min_z, z <= max_z)
        is_healthy = jnp.logical_and(is_finite, z_ok)
        return is_healthy

    def state_info(self, state: mjx.Data, params: AntMJXEnvParams) -> dict[str, float]:
        """Includes state information exclueded from `observation()`."""
        mjx_data = state

        info = {
            "x_position": mjx_data.qpos[0],
            "y_position": mjx_data.qpos[1],
            "distance_from_origin": jnp.linalg.norm(mjx_data.qpos[0:2], ord=2),
        }
        return info

    def terminal(
        self, state: mjx.Data, rng: jax.Array, params: AntMJXEnvParams
    ) -> bool:
        """Terminates if unhealthy."""
        return jnp.logical_and(
            jnp.logical_not(self._gen_is_healthy(state, params)),
            params.terminate_when_unhealthy,
        )

    def get_default_params(self, **kwargs) -> AntMJXEnvParams:
        """Get the default parameter for the Ant environment."""
        base_params = super().get_default_params()
        return AntMJXEnvParams(
            xml_file=kwargs.get("xml_file", 1),
            frame_skip=kwargs.get("frame_skip", 5),
            default_camera_config=kwargs.get("default_camera_config", DEFAULT_CAMERA_CONFIG),
            forward_reward_weight=kwargs.get("forward_reward_weight", 1),
            ctrl_cost_weight=kwargs.get("ctrl_cost_weight", 0.5),
            contact_cost_weight=kwargs.get("contact_cost_weight", 5e-4),
            healthy_reward=kwargs.get("healthy_reward", 1.0),
            main_body=kwargs.get("main_body", 1),
            terminate_when_unhealthy=kwargs.get("terminate_when_unhealthy", True),
            healthy_z_range=kwargs.get("healthy_z_range", (0.2, 1.0)),
            contact_force_range=kwargs.get("contact_force_range", (-1.0, 1.0)),
            reset_noise_scale=kwargs.get("reset_noise_scale", 0.1),
            exclude_current_positions_from_observation=kwargs.get("exclude_current_positions_from_observation", True),
            include_cfrc_ext_in_observation=kwargs.get("include_cfrc_ext_in_observation", True),
            camera_id=kwargs.get("camera_id", base_params.camera_id),
            camera_name=kwargs.get("camera_name", base_params.camera_name),
            max_geom=kwargs.get("max_geom", base_params.max_geom),
            width=kwargs.get("width", base_params.width),
            height=kwargs.get("height", base_params.height),
            render_mode=kwargs.get("render_mode", base_params.render_mode),
        )


class AntJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based Ant environment using the MJX functional implementation as base."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(self, render_mode: str | None = None, **kwargs: any):
        """Constructor where the kwargs are passed to the base environment to modify the parameters."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        # Get default params and apply kwargs overrides
        temp_env = Ant_MJXEnv()
        params = temp_env.get_default_params(**kwargs)

        env = Ant_MJXEnv(params=params)
        env.transform(jax.jit)

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )