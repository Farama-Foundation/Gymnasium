"""Contains the classes for the manipulation environments, `Pusher`, `Reacher`."""
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
from gymnasium.envs.mujoco.pusher_v5 import (
    DEFAULT_CAMERA_CONFIG as PUSHER_DEFAULT_CAMERA_CONFIG,
)
from gymnasium.envs.mujoco.reacher_v5 import (
    DEFAULT_CAMERA_CONFIG as REACHER_HOPPER_DEFAULT_CAMERA_CONFIG,
)


class Reacher_MJXEnv(MJXEnv):
    """Class for Reacher."""

    def __init__(
        self,
        params: Dict[str, any],
    ):
        """Sets the `obveration_space`."""
        MJXEnv.__init__(self, params=params)

        self.observation_space = gymnasium.spaces.Box(  # TODO use jnp when and if `Box` supports jax natively
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def _gen_init_physics_state(
        self, rng, params: Dict[str, any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sets `arm.qpos` (positional elements) and `arm.qvel` (velocity elements) from a CUD and the `goal.qpos` from a cicrular uniform distribution."""
        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            key=rng, minval=-0.1, maxval=0.1, shape=(self.mjx_model.nq,)
        )

        while True:
            goal = jax.random.uniform(key=rng, minval=-0.2, maxval=0.2, shape=(2,))
            c_bool = jnp.less(jnp.linalg.norm(goal), jnp.array(0.2))
            c_bool = jnp.less(jnp.linalg.norm(jnp.array([-0.15, 0.1])), 0.2)
            #breakpoint()
            #if c_bool:
                #break
            # if jnp.less(jnp.linalg.norm(goal), jnp.array(0.2)):
                # break
            # TODO FIX THIS
            if True:
                break
        qpos.at[-2:].set(goal)

        qvel = jax.random.uniform(
            key=rng, minval=-0.005, maxval=0.005, shape=(self.mjx_model.nv,)
        )
        qvel.at[-2:].set(jnp.zeros(2))

        act = jnp.empty(self.mjx_model.na)

        return qpos, qvel, act

    def _get_goal(self, mjx_data: mjx.Data) -> jnp.ndarray:
        return mjx_data.qpos[-2:]

    def _set_goal(self, mjx_data: mjx.Data, goal: jnp.ndarray) -> mjx.Data:
        """Add the coordinate of `goal` to `mjx_data`."""
        mjx_data = mjx_data.replace(qpos=mjx_data.qpos.at[-2:].set(goal))
        return mjx_data

    def observation(
        self, state: mjx.Data, rng: jax.random.PRNGKey, params: Dict[str, any]
    ) -> jnp.ndarray:
        """Observes the `sin(theta)` & `cos(theta)` & `qpos` &  `qvel` & 'fingertip - target' distance."""
        mjx_data = state

        position = mjx_data.qpos.flatten()
        velocity = mjx_data.qvel.flatten()
        theta = position[:2]

        fingertip_position = mjx_data.xpos[3]  # TODO make this dynamic
        target_position = mjx_data.xpos[4]  # TODO make this dynamic
        observation = jnp.concatenate(
            (
                jnp.cos(theta),
                jnp.sin(theta),
                position[2:],
                velocity[:2],
                (fingertip_position - target_position)[:2],
            )
        )

        return observation

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: Dict[str, any],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Reward = reward_dist + reward_ctrl."""
        mjx_data = next_state

        fingertip_position = mjx_data.xpos[3]  # TODO make this dynamic
        target_position = mjx_data.xpos[4]  # TODO make this dynamic

        vec = fingertip_position - target_position
        reward_dist = -jnp.linalg.norm(vec) * params["reward_dist_weight"]
        reward_ctrl = -jnp.square(action).sum() * params["reward_control_weight"]

        reward = reward_dist + reward_ctrl

        reward_info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
        }

        return reward, reward_info

    def get_default_params(**kwargs) -> Dict[str, any]:
        """Get the default parameter for the Reacher environment."""
        default = {
            "xml_file": "reacher.xml",
            "frame_skip": 2,
            "default_camera_config": REACHER_HOPPER_DEFAULT_CAMERA_CONFIG,
            "reward_dist_weight": 1,
            "reward_control_weight": 1,
        }
        return {**MJXEnv.get_default_params(), **default, **kwargs}


class Pusher_MJXEnv(MJXEnv):
    # NOTE: MJX does not yet support condim=1 and therefore this class can not be instantiated
    """Class for Pusher."""

    def __init__(
        self,
        params: Dict[str, any],
    ):
        """Sets the `obveration_space`."""
        MJXEnv.__init__(self, params=params)

        self.observation_space = gymnasium.spaces.Box(  # TODO use jnp when and if `Box` supports jax natively
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )

    def _gen_init_physics_state(
        self, rng, params: Dict[str, any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sets `arm.qpos` (positional elements) and `arm.qvel` (velocity elements) from a CUD and the `goal.qpos` from a cicrular uniform distribution."""
        qpos = self.mjx_model.qpos0

        goal_pos = jnp.zeroes(2)
        while True:
            cylinder_pos = np.concatenate(
                [
                    jax.random.uniform(key=rng, minval=-0.3, maxval=0.3, shape=1),
                    jax.random.uniform(key=rng, minval=-0.2, maxval=0.2, shape=1),
                ]
            )
            if jnp.linalg.norm(cylinder_pos - goal_pos) > 0.17:
                break

        qpos.at[-4:-2].set(cylinder_pos)
        qpos.at[-2:].set(goal_pos)
        qvel = jax.random.uniform(
            key=rng, minval=-0.005, maxval=0.005, shape=(self.mjx_model.nv,)
        )
        qvel.at[-4:].set(0)

        act = jnp.empty(self.mjx_model.na)

        return qpos, qvel, act

    def observation(
        self, state: mjx.Data, rng: jax.random.PRNGKey, params: Dict[str, any]
    ) -> jnp.ndarray:
        """Observes the & `qpos` &  `qvel` & `tips_arm` & `object` `goal`."""
        mjx_data = state

        position = mjx_data.qpos.flatten()
        velocity = mjx_data.qvel.flatten()
        tips_arm_position = mjx_data.xpos[10]  # TODO make this dynamic
        object_position = mjx_data.xpos[11]  # TODO make this dynamic
        goal_position = mjx_data.xpos[12]  # TODO make this dynamic

        observation = jnp.concatenate(
            (
                position[:7],
                velocity[:7],
                tips_arm_position,
                object_position,
                goal_position,
            )
        )

        return observation

    def _get_reward(
        self,
        state: mjx.Data,
        action: jnp.ndarray,
        next_state: mjx.Data,
        params: Dict[str, any],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Reward = reward_dist + reward_ctrl + reward_near."""
        mjx_data = next_state
        tips_arm_position = mjx_data.xpos[10]  # TODO make this dynamic
        object_position = mjx_data.xpos[11]  # TODO make this dynamic
        goal_position = mjx_data.xpos[12]  # TODO make this dynamic

        vec_1 = object_position - tips_arm_position
        vec_2 = object_position - goal_position

        reward_near = -jnp.linalg.norm(vec_1) * params["reward_near_weight"]
        reward_dist = -jnp.linalg.norm(vec_2) * params["reward_dist_weight"]
        reward_ctrl = -jnp.square(action).sum() * params["reward_control_weight"]

        reward = reward_dist + reward_ctrl + reward_near

        reward_info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_near": reward_near,
        }

        return reward, reward_info

    def get_default_params(**kwargs) -> Dict[str, any]:
        """Get the default parameter for the Reacher environment."""
        default = {
            "xml_file": "pusher.xml",
            "frame_skip": 5,
            "default_camera_config": PUSHER_DEFAULT_CAMERA_CONFIG,
            "reward_near_weight": 0.5,
            "reward_dist_weight": 1,
            "reward_control_weight": 0.1,
        }
        return {**MJXEnv.get_default_params(), **default, **kwargs}
