import typing
from typing import Optional, Union

import numpy as np

import gymnasium
from gymnasium import Space
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Discrete


class TestingEnv(gymnasium.Env):
    """A testing environment for wrappers where a custom action or observation can be passed to the environment.
    The action and observation spaces provided are used to sample new observations or actions to test with the environment
    """

    __test__ = False

    def __init__(
        self,
        observation_space: Space = None,
        action_space: Space = None,
        reward_range=None,
        env_length: Optional[int] = None,
    ):
        """Constructor of the testing environment
        Args:
            observation_space: The environment observation shape
            action_space: The environment action space
            reward_range: The reward range for the environment to sample from
            env_length: The environment length used to know if the environment has timed out
        """
        self.name = ""
        if observation_space is None:
            self.observation_space = Box(-10, 10, ())
        else:
            self.observation_space = observation_space
            self.name += str(observation_space)

        if action_space is None:
            self.action_space = Discrete(5)
        else:
            self.action_space = action_space
            self.name += str(action_space)

        if reward_range is None:
            self.reward_range = (0, 1)
        else:
            self.reward_range = reward_range
            self.name += str(reward_range)

        self.env_length = env_length
        self.steps_left = env_length

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Union[ObsType, typing.Tuple[ObsType, dict]]:
        """Reset the environment."""
        self.steps_left = self.env_length
        return self.observation_space.sample(), {}

    def step(self, action: ActType) -> typing.Tuple[ObsType, float, bool, dict]:
        """Step through the environment."""
        if self.env_length is not None:
            self.steps_left -= 1

        return (
            self.observation_space.sample(),
            np.random.randint(self.reward_range[0], self.reward_range[1]),
            False,  # terminated currently not handled
            self.env_length is not None and self.steps_left == 0,
            {"action": action},
        )

    def __str__(self):
        return self.name
