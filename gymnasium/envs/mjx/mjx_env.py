"""Contains the base class for MJX based robot environments.

Note: This is expted to be used my `gymnasium`, `gymnasium-robotics`, `metaworld` and 3rd party libraries.
"""
from typing import Dict, Tuple, Union

import numpy as np

import gymnasium
from gymnasium.envs.mujoco import MujocoRenderer
from gymnasium.envs.mujoco.mujoco_env import expand_model_path


try:
    import jax
    import mujoco
    from jax import numpy as jnp
    from mujoco import mjx
except ImportError as e:
    MJX_IMPORT_ERROR = e
else:
    MJX_IMPORT_ERROR = None


# state = np.empty(mujoco.mj_stateSize(env.unwrapped.model, mujoco.mjtState.mjSTATE_PHYSICS))
# mujoco.mj_getState(env.unwrapped.model, env.unwrapped.data, state, spec=mujoco.mjtState.mjSTATE_PHYSICS)

# mujoco.mj_setState(env.unwrapped.model, env.unwrapped.data, state, spec=mujoco.mjtState.mjSTATE_PHYSICS)


"""
# TODO unit test these
def mjx_get_physics_state(mjx_data: mjx.Data) -> jnp.ndarray:
    ""Get physics state of `mjx_data` similar to mujoco.get_state.""
    return jnp.concatenate([mjx_data.qpos, mjx_data.qvel, mjx_data.act])


def mjx_set_physics_state(mjx_data: mjx.Data, mjx_physics_state) -> mjx.Data:
    ""Sets the physics state in `mjx_data`.""
    qpos_end_index = mjx_data.qpos.size
    qvel_end_index = qpos_end_index + mjx_data.qvel.size

    qpos = mjx_physics_state[:qpos_end_index]
    qvel = mjx_physics_state[qpos_end_index: qvel_end_index]
    act = mjx_physics_state[qvel_end_index:]
    assert qpos.size == mjx_data.qpos.size
    assert qvel.size == mjx_data.qvel.size
    assert act.size == mjx_data.act.size

    return mjx_data.replace(qpos=qpos, qvel=qvel, act=act)
"""


# TODO add type hint to `params`
# TODO add render `metadata`
# TODO add init_qvel
class MJXEnv(
    gymnasium.functional.FuncEnv[
        mjx.Data,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        bool,
        MujocoRenderer,
        Dict[str, any],
    ]
):
    """The Base class for MJX Environments in Gymnasium.

    `observation`, `terminal`, and `state_info` should be defined in sub-classes.
    """

    def __init__(self, params: Dict[str, any]):
        """Create the `mjx.Model` of the robot defined in `params["xml_file"]`.

        Keep `mujoco.MjModel` of model for rendering purposes.
        The Sub-class environments are expected to define `self.observation_space`
        """
        if MJX_IMPORT_ERROR is not None:
            raise gymnasium.error.DependencyNotInstalled(
                f"{MJX_IMPORT_ERROR}. "
                "(HINT: you need to install mujoco-mjx, run `pip install gymnasium[mjx]`.)"  # TODO actually create gymnasium[mjx]
            )

        fullpath = expand_model_path(params["xml_file"])

        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.mjx_model = mjx.put_model(self.model)

        # observation_space: gymnasium.spaces.Box  # set by subclass
        self.action_space = gymnasium.spaces.Box(
            low=self.model.actuator_ctrlrange.T[0],
            high=self.model.actuator_ctrlrange.T[1],
            dtype=np.float32,
        )
        # TODO change bounds and types when and if `Box` supports JAX nativly
        # self.action_space = gymnasium.spaces.Box(low=self.mjx_model.actuator_ctrlrange.T[0], high=self.mjx_model.actuator_ctrlrange.T[1], dtype=np.float32)

    def initial(self, rng: jax.random.PRNGKey, params: Dict[str, any]) -> mjx.Data:
        """Initializes and returns the `mjx.Data`."""
        # TODO? find a more performant alternative that does not allocate?
        mjx_data = mjx.make_data(self.model)
        qpos, qvel, act = self._gen_init_physics_state(rng, params)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, act=act)
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        return mjx_data

    def transition(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        rng: jax.random.PRNGKey,
        params: Dict[str, any],
    ) -> mjx.Data:
        """Step through the simulator using `action` for `self.dt` (note: `rng` argument is ignored)."""
        mjx_data = state

        mjx_data = mjx_data.replace(ctrl=action)
        mjx_data = jax.lax.fori_loop(
            0, params["frame_skip"], lambda _, x: mjx.step(self.mjx_model, x), mjx_data
        )

        # TODO fix sensors with MJX>=3.2
        return mjx_data

    def reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        rng: jax.random.PRNGKey,
        params: Dict[str, any],
    ) -> jnp.ndarray:
        """Returns the reward."""
        return self._get_reward(state, action, next_state, params)[0]

    def transition_info(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: Dict[str, any],
    ) -> Dict:
        """Includes just reward info."""
        return self._get_reward(state, action, next_state, params)[1]

    def render_init(
        self,
        params: Dict[str, any],
    ) -> MujocoRenderer:
        """Returns a `MujocoRenderer` object."""
        return MujocoRenderer(
            self.model,
            None,  # no MuJoCo DATA
            params["default_camera_config"],
            params["width"],
            params["height"],
            params["max_geom"],
            params["camera_id"],
            params["camera_name"],
        )

    def render_image(
        self,
        state: mjx.Data,
        render_state: MujocoRenderer,
        params: Dict[str, any],
    ) -> Tuple[MujocoRenderer, Union[np.ndarray, None]]:
        """Renders the `mujoco` frame of the environment by converting `mjx.Data` to `mujoco.MjData`.

        NOTE: this function can not be jitted.
        """
        mjx_data = state
        mujoco_renderer = render_state

        data = mjx.get_data(self.model, mjx_data)
        mujoco.mj_forward(self.model, data)

        mujoco_renderer.data = data

        frame = mujoco_renderer.render(params["render_mode"])

        return mujoco_renderer, frame

    def render_close(
        self, render_state: MujocoRenderer, params: Dict[str, any]
    ) -> None:
        """Closes the `MujocoRender` object."""
        mujoco_renderer = render_state
        if mujoco_renderer is not None:
            mujoco_renderer.close()

    def dt(self, params: Dict[str, any]) -> float:
        """Returns the duration between timesteps (`dt`)."""
        return self.mjx_model.opt.timestep * params["frame_skip"]

    def _gen_init_physics_state(
        self, rng, params: Dict[str, any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generates the initial physics state.

        `MJXEnv` Equivalent of `MujocoEnv.model.`

        Returns: `(qpos, qvel, act)`
        """
        raise NotImplementedError

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: Dict[str, any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """Generates `reward` and `transition_info`, we rely on the JIT's SEE to optimize it.

        Returns: `(reward, reward_info)`
        """
        raise NotImplementedError

    def terminal(
        self,
        state: mjx.Data,
        rng: jax.random.PRNGKey,
        params: Dict[str, any] | None = None,
    ) -> jnp.ndarray:
        """Should be overwritten if the sub-class environment terminates."""
        return jnp.array(False)

    def get_default_params(**kwargs) -> Dict[str, any]:
        """Generate the default parameters for rendering."""
        default = {
            "default_camera_config": {},
            "camera_id": None,
            "camera_name": None,
            "max_geom": 1000,
            "width": 480,
            "height": 480,
            "render_mode": None,
        }
        return default

    """
    def mjx_get_physics_state_put_version(self, mjx_data: mjx.Data) -> np.ndarray:
        ""version based on @btaba suggestion""
        # data = mujoco.MjData(self.model)
        # mjx.device_get_into(data, mjx_data)
        data = mjx.get_data(self.model, mjx_data)
        state = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_PHYSICS))
        mujoco.mj_getState(self.model, data, state, spec=mujoco.mjtState.mjSTATE_PHYSICS)

        return state
    """


# TODO add vector environment
# TODO consider requirement of `metaworld` & `gymansium_robotics.RobotEnv` & `mo-gymnasium`
# TODO unit testing
