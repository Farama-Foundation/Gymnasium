"""A collection of wrappers for modifying the reward.

* ``TransformReward`` - Transforms the reward by a function
* ``ClipReward`` - Clips the reward between a minimum and maximum value
"""

from __future__ import annotations

from collections.abc import Callable
from typing import SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.error import InvalidBound


__all__ = ["TransformReward", "ClipReward"]


class TransformReward(
    gym.RewardWrapper[ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Applies a function to the ``reward`` received from the environment's ``step``.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.TransformReward`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformReward
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformReward(env, lambda r: 2 * r + 1)
        >>> _ = env.reset()
        >>> _, rew, _, _, _ = env.step(0)
        >>> rew
        3.0

    Change logs:
     * v0.15.0 - Initially added
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[SupportsFloat], SupportsFloat],
    ):
        """Initialize TransformReward wrapper.

        Args:
            env (Env): The environment to wrap
            func: (Callable): The function to apply to reward
        """
        gym.utils.RecordConstructorArgs.__init__(self, func=func)
        gym.RewardWrapper.__init__(self, env)

        self.func = func

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Apply function to reward.

        Args:
            reward (Union[float, int, np.ndarray]): environment's reward
        """
        return self.func(reward)


class ClipReward(TransformReward[ObsType, ActType], gym.utils.RecordConstructorArgs):
    """Clips the rewards for an environment between an upper and lower bound.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.ClipReward`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ClipReward
        >>> env = gym.make("CartPole-v1")
        >>> env = ClipReward(env, 0, 0.5)
        >>> _ = env.reset()
        >>> _, rew, _, _, _ = env.step(1)
        >>> rew
        np.float64(0.5)

    Change logs:
     * v1.0.0 - Initially added
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        min_reward: float | np.ndarray | None = None,
        max_reward: float | np.ndarray | None = None,
    ):
        """Initialize ClipRewards wrapper.

        Args:
            env (Env): The environment to wrap
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

        gym.utils.RecordConstructorArgs.__init__(
            self, min_reward=min_reward, max_reward=max_reward
        )
        TransformReward.__init__(
            self, env=env, func=lambda x: np.clip(x, a_min=min_reward, a_max=max_reward)
        )
