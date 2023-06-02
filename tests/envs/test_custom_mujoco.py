__credits__ = ["Kallinteris-Andreas"]

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import gymansium as gym
from gymnasium.spaces import Box
import pytest
from gymnasium.error import Error
import warnings


class PointEnv(MujocoEnv, utils.EzPickle):
    """
    A simple mujuco env to test thrid party mujoco env, using the `Gymansium.MujocoEnv` environment API.
    """

    def __init__(
        self,
        xml_file="point.xml",
        frame_skip=1,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            **kwargs
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            # "render_fps": 100 / frame_skip,  # do not compute it
        }

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config={},
            **kwargs
        )

        obs_size = self.data.qpos.size + self.data.qvel.size

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]

        observation = self._get_obs()
        reward = x_position_after - x_position_before
        info = {}

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        return np.concatenate((position, velocity))

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
    ]
]


@pytest.mark.parametrize("frame_skip", [1, 2, 3, 4, 5])
def test_frame_skip(frame_skip):
    """verify that cusstom envs work with different `frame_skip` values"""
    env = PointEnv(frame_skip=frame_skip)

    # Test if env adheres to Gym API
    with warnings.catch_warnings(record=True) as w:
        gym.utils.env_checker.check_env(env.unwrapped, skip_render_check=True)
        env.close()
    for warning in w:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning.message}")

