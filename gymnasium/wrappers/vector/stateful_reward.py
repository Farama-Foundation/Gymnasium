"""A collection of wrappers for modifying the reward with an internal state.

* ``NormalizeReward`` - Normalizes the rewards to a mean and standard deviation
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.vector.vector_env import ArrayType, VectorEnv, VectorWrapper
from gymnasium.wrappers.utils import RunningMeanStd


__all__ = ["NormalizeReward"]


class NormalizeReward(VectorWrapper, gym.utils.RecordConstructorArgs):
    r"""This wrapper will scale rewards s.t. the discounted returns have a mean of 0 and std of 1.

    In a nutshell, the rewards are divided through by the standard deviation of a rolling discounted sum of the reward.
    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the reward
    statistics. If `True` (default), the `RunningMeanStd` will get updated every time `self.normalize()` is called.
    If False, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Important note:
        Contrary to what the name suggests, this wrapper does not normalize the rewards to have a mean of 0 and a standard
        deviation of 1. Instead, it scales the rewards such that **discounted returns** have approximately unit variance.
        See [Engstrom et al.](https://openreview.net/forum?id=r1etN1rtPB) on "reward scaling" for more information.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.

    Example without the normalize reward wrapper:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> envs = gym.make_vec("MountainCarContinuous-v0", 3)
        >>> _ = envs.reset(seed=123)
        >>> _ = envs.action_space.seed(123)
        >>> episode_rewards = []
        >>> for _ in range(100):
        ...     observation, reward, *_ = envs.step(envs.action_space.sample())
        ...     episode_rewards.append(reward)
        ...
        >>> envs.close()
        >>> np.mean(episode_rewards)
        np.float64(-0.03359492141887935)
        >>> np.std(episode_rewards)
        np.float64(0.029028230434438706)

    Example with the normalize reward wrapper:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> envs = gym.make_vec("MountainCarContinuous-v0", 3)
        >>> envs = NormalizeReward(envs)
        >>> _ = envs.reset(seed=123)
        >>> _ = envs.action_space.seed(123)
        >>> episode_rewards = []
        >>> for _ in range(100):
        ...     observation, reward, *_ = envs.step(envs.action_space.sample())
        ...     episode_rewards.append(reward)
        ...
        >>> envs.close()
        >>> np.mean(episode_rewards)
        np.float64(-0.1598639586606745)
        >>> np.std(episode_rewards)
        np.float64(0.27800309628058434)
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
        VectorWrapper.__init__(self, env)

        self.return_rms = RunningMeanStd(shape=())
        self.accumulated_reward: np.array = np.zeros((self.num_envs,), dtype=np.float32)
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
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = super().step(actions)
        self.accumulated_reward = (
            self.accumulated_reward * self.gamma * (1 - terminated) + reward
        )
        return obs, self.normalize(reward), terminated, truncated, info

    def normalize(self, reward: SupportsFloat):
        """Normalizes the rewards with the running mean rewards and their variance."""
        if self._update_running_mean:
            self.return_rms.update(self.accumulated_reward)
        return reward / np.sqrt(self.return_rms.var + self.epsilon)
