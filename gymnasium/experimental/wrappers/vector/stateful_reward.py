"""A collection of wrappers for modifying the reward with an internal state.

* ``NormalizeRewardV0`` - Normalizes the rewards to a mean and standard deviation
"""
from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.experimental.vector.vector_env import VectorEnv, VectorWrapper
from gymnasium.experimental.wrappers.utils import RunningMeanStd


__all__ = ["NormalizeRewardV1"]


class NormalizeRewardV1(VectorWrapper, gym.utils.RecordConstructorArgs):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the reward
    statistics. If `True` (default), the `RunningMeanStd` will get updated every time `self.normalize()` is called.
    If False, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: VectorEnv,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        self.rewards_running_means = RunningMeanStd(shape=())
        self.discounted_reward: np.array = np.zeros((self.num_envs,), dtype=np.float32)
        self.gamma = gamma
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the reward statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the reward statistics."""
        self._update_running_mean = setting

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = super().step(action)
        self.discounted_reward = (
            self.discounted_reward * self.gamma * (1 - terminated) + reward
        )
        return obs, self.normalize(reward), terminated, truncated, info

    def normalize(self, reward: SupportsFloat):
        """Normalizes the rewards with the running mean rewards and their variance."""
        if self._update_running_mean:
            self.rewards_running_means.update(self.discounted_reward)
        return reward / np.sqrt(self.rewards_running_means.var + self.epsilon)
