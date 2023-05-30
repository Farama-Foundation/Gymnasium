"""Vectorizes reward function to work with `VectorEnv`."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from gymnasium import Env
from gymnasium.experimental.vector import VectorEnv, VectorRewardWrapper
from gymnasium.experimental.vector.vector_env import ArrayType
from gymnasium.experimental.wrappers import lambda_reward


class LambdaRewardV0(VectorRewardWrapper):
    """A reward wrapper that allows a custom function to modify the step reward."""

    def __init__(self, env: VectorEnv, func: Callable[[ArrayType], ArrayType]):
        """Initialize LambdaRewardV0 wrapper.

        Args:
            env (Env): The vector environment to wrap
            func: (Callable): The function to apply to reward
        """
        super().__init__(env)

        self.func = func

    def reward(self, reward: ArrayType) -> ArrayType:
        """Apply function to reward."""
        return self.func(reward)


class VectorizeLambdaRewardV0(VectorRewardWrapper):
    """Vectorizes a single-agent lambda reward wrapper for vector environments."""

    def __init__(
        self, env: VectorEnv, wrapper: type[lambda_reward.LambdaRewardV0], **kwargs: Any
    ):
        """Constructor for the vectorized lambda reward wrapper.

        Args:
            env: The vector environment to wrap.
            wrapper: The wrapper to vectorize
            **kwargs: Keyword argument for the wrapper
        """
        super().__init__(env)

        self.wrapper = wrapper(Env(), **kwargs)

    def reward(self, reward: ArrayType) -> ArrayType:
        """Iterates over the reward updating each with the wrapper func."""
        for i, r in enumerate(reward):
            reward[i] = self.wrapper.func(r)
        return reward


class ClipRewardV0(VectorizeLambdaRewardV0):
    """A wrapper that clips the rewards for an environment between an upper and lower bound."""

    def __init__(
        self,
        env: VectorEnv,
        min_reward: float | np.ndarray | None = None,
        max_reward: float | np.ndarray | None = None,
    ):
        """Constructor for ClipReward wrapper.

        Args:
            env: The vector environment to wrap
            min_reward: The min reward for each step
            max_reward: the max reward for each step
        """
        super().__init__(
            env,
            lambda_reward.ClipRewardV0,
            min_reward=min_reward,
            max_reward=max_reward,
        )
