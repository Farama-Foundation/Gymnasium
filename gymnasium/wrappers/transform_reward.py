"""Wrapper for transforming the reward."""
from typing import Callable

import gymnasium as gym
from gymnasium import RewardWrapper


class TransformReward(RewardWrapper):
    """Transform the reward via an arbitrary function.

    Warning:
        If the base environment specifies a reward range which is not invariant under :attr:`f`, the :attr:`reward_range` of the wrapped environment will be incorrect.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformReward
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformReward(env, lambda r: 0.01*r)
        >>> _ = env.reset()
        >>> observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        >>> reward
        0.01
    """

    def __init__(self, env: gym.Env, f: Callable[[float], float]):
        """Initialize the :class:`TransformReward` wrapper with an environment and reward transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the reward
        """
        super().__init__(env)
        assert callable(f)
        self.f = f

    def reward(self, reward):
        """Transforms the reward using callable :attr:`f`.

        Args:
            reward: The reward to transform

        Returns:
            The transformed reward
        """
        return self.f(reward)
