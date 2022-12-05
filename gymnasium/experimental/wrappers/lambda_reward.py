"""A collection of wrappers for modifying the reward.

* ``LambdaReward`` - Transforms the reward by a function
* ``ClipReward`` - Clips the reward between a minimum and maximum value
"""

from __future__ import annotations

from typing import Callable, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.error import InvalidBound


class LambdaRewardV0(gym.RewardWrapper):
    """A reward wrapper that allows a custom function to modify the step reward.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import LambdaRewardV0
        >>> env = gym.make("CartPole-v1")
        >>> env = LambdaRewardV0(env, lambda r: 2 * r + 1)
        >>> _ = env.reset()
        >>> _, rew, _, _, _ = env.step(0)
        >>> rew
        3.0
    """

    def __init__(
        self,
        env: gym.Env,
        func: Callable[[SupportsFloat], SupportsFloat],
    ):
        """Initialize LambdaRewardV0 wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            func: (Callable): The function to apply to reward
        """
        super().__init__(env)

        self.func = func

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Apply function to reward.

        Args:
            reward (Union[float, int, np.ndarray]): environment's reward
        """
        return self.func(reward)


class ClipRewardV0(LambdaRewardV0):
    """A wrapper that clips the rewards for an environment between an upper and lower bound.

    Example with an upper and lower bound:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import ClipRewardV0
        >>> env = gym.make("CartPole-v1")
        >>> env = ClipRewardV0(env, 0, 0.5)
        >>> env.reset()
        >>> _, rew, _, _, _ = env.step(1)
        >>> rew
        0.5
    """

    def __init__(
        self,
        env: gym.Env,
        min_reward: float | np.ndarray | None = None,
        max_reward: float | np.ndarray | None = None,
    ):
        """Initialize ClipRewardsV0 wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_reward (Union[float, np.ndarray]): lower bound to apply
            max_reward (Union[float, np.ndarray]): higher bound to apply
        """
        if min_reward is None and max_reward is None:
            raise InvalidBound("Both `min_reward` and `max_reward` cannot be None")

        elif max_reward is not None and min_reward is not None:
            if np.any(max_reward - min_reward < 0):
                raise InvalidBound(
                    f"Min reward ({min_reward}) must be smaller than max reward ({max_reward})"
                )

        super().__init__(env, lambda x: np.clip(x, a_min=min_reward, a_max=max_reward))
