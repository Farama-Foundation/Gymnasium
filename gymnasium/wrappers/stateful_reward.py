"""A collection of wrappers for modifying the reward with an internal state.

* ``NormalizeRewardV1`` - Normalizes the rewards to a mean and standard deviation
"""
from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers.utils import RunningMeanStd


__all__ = ["NormalizeRewardV1"]


class NormalizeRewardV1(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    r"""Normalizes immediate rewards such that their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the reward
    statistics. If `True` (default), the `RunningMeanStd` will get updated every time `self.normalize()` is called.
    If False, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Note:
        In v0.27, NormalizeReward was updated as the forward discounted reward estimate was incorrectly computed in Gym v0.25+.
        For more detail, read [#3154](https://github.com/openai/gym/pull/3152).

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.

    Example:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> env = gym.make('MountainCarContinuous-v0')
        >>> _ = env.reset(seed=123)
        >>> _ = env.action_space.seed(123)
        >>> episode_rewards = []
        >>> terminated, truncated = False, False
        >>> while not (terminated or truncated):
        ...     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ...     episode_rewards.append(reward)
        ...
        >>> np.var(episode_rewards)
        0.0008876301247721108

        >>> env = gym.make('MountainCarContinuous-v0')
        >>> env = NormalizeRewardV1(env, gamme=0.99, epsilon=1e-8)
        >>> _ = env.reset(seed=123)
        >>> _ = env.action_space.seed(123)
        >>> episode_rewards = []
        >>> terminated, truncated = False, False
        >>> while not (terminated or truncated):
        ...     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ...     episode_rewards.append(reward)
        ...
        >>> # will approach 0.99 with more episodes
        >>> np.var(episode_rewards)
        0.010162116476634746
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
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

        self.return_rms = RunningMeanStd(shape=())
        self.discounted_reward: np.array = np.array([0.0])
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
        self.discounted_reward = self.discounted_reward * self.gamma * (
            1 - terminated
        ) + float(reward)
        return obs, self.normalize(float(reward)), terminated, truncated, info

    def normalize(self, reward: SupportsFloat):
        """Normalizes the rewards with the running mean rewards and their variance."""
        if self._update_running_mean:
            self.return_rms.update(self.discounted_reward)
        return reward / np.sqrt(self.return_rms.var + self.epsilon)
