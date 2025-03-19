"""A collection of wrappers for modifying the reward with an internal state.

* ``NormalizeReward`` - Normalizes the rewards to a mean and standard deviation
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers.utils import RunningMeanStd


__all__ = ["NormalizeReward"]


class NormalizeReward(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    r"""Normalizes immediate rewards such that their exponential moving average has an approximately fixed variance.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the reward
    statistics. If `True` (default), the `RunningMeanStd` will get updated every time `self.normalize()` is called.
    If False, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.NormalizeReward`.

    Note:
        In v0.27, NormalizeReward was updated as the forward discounted reward estimate was incorrectly computed in Gym v0.25+.
        For more detail, read [#3154](https://github.com/openai/gym/pull/3152).

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.

    Example without the normalize reward wrapper:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> env = gym.make("MountainCarContinuous-v0")
        >>> _ = env.reset(seed=123)
        >>> _ = env.action_space.seed(123)
        >>> episode_rewards = []
        >>> terminated, truncated = False, False
        >>> while not (terminated or truncated):
        ...     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ...     episode_rewards.append(reward)
        ...
        >>> env.close()
        >>> np.var(episode_rewards)
        np.float64(0.0008876301247721108)

    Example with the normalize reward wrapper:
        >>> import numpy as np
        >>> import gymnasium as gym
        >>> env = gym.make("MountainCarContinuous-v0")
        >>> env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
        >>> _ = env.reset(seed=123)
        >>> _ = env.action_space.seed(123)
        >>> episode_rewards = []
        >>> terminated, truncated = False, False
        >>> while not (terminated or truncated):
        ...     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ...     episode_rewards.append(reward)
        ...
        >>> env.close()
        >>> np.var(episode_rewards)
        np.float64(0.010162116476634746)

    Change logs:
     * v0.21.0 - Initially added
     * v1.0.0 - Add `update_running_mean` attribute to allow disabling of updating the running mean / standard
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has an approximately fixed variance.

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

        # Using the `discounted_reward` rather than `reward` makes no sense but for backward compatibility, it is being kept
        self.discounted_reward = self.discounted_reward * self.gamma * (
            1 - terminated
        ) + float(reward)
        if self._update_running_mean:
            self.return_rms.update(self.discounted_reward)

        # We don't (reward - self.return_rms.mean) see https://github.com/openai/baselines/issues/538
        normalized_reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
        return obs, normalized_reward, terminated, truncated, info
