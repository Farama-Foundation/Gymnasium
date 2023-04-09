"""A collection of wrappers for modifying the reward.

* ``LambdaRewardV0`` - Transforms the reward by a function
* ``VectoriseLambdaRewardV0`` - Vectorises a lambda reward wrapper
* ``ClipRewardV0`` - Clips the reward between a minimum and maximum value
"""
from __future__ import annotations

from typing import Any, Callable

from gymnasium import Env
from gymnasium.experimental import wrappers
from gymnasium.experimental.vector import VectorEnv, VectorRewardWrapper
from gymnasium.experimental.vector.vector_env import ArrayType


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


class VectoriseLambdaRewardV0(VectorRewardWrapper):
    """Vectorises a single-agent lambda reward wrapper for vector environments."""

    def __init__(
        self, env: VectorEnv, wrapper: type[wrappers.LambdaRewardV0], **kwargs: Any
    ):
        """Constructor for the vectorised lambda reward wrapper.

        Args:
            env: The vector environment to wrap.
            wrapper: The wrapper to vectorise
            **kwargs: Keyword argument for the wrapper
        """
        super().__init__(env)

        self.wrapper = wrapper(Env(), **kwargs)

    def reward(self, reward: ArrayType) -> ArrayType:
        """Iterates over the reward updating each with the wrapper func."""
        for i, r in enumerate(reward):
            reward[i] = self.wrapper.func(r)
        return reward


class ClipRewardV0(VectoriseLambdaRewardV0):
    """A wrapper that clips the rewards for an environment between an upper and lower bound."""

    def __init__(self, env: VectorEnv):
        """Constructor for ClipReward wrapper.

        Args:
            env: The vector environment to wrap
        """
        super().__init__(env, wrappers.ClipRewardV0)
