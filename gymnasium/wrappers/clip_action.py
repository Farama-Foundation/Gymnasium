"""Wrapper for clipping actions within a valid bound."""
import numpy as np

import gymnasium as gym
from gymnasium import ActionWrapper
from gymnasium.spaces import Box


class ClipAction(ActionWrapper):
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ClipAction
        >>> env = gym.make("Hopper-v4")
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-1.0, 1.0, (3,), float32)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([5.0, -2.0, 0.0]))
        ... # Executes the action np.array([1.0, -1.0, 0]) in the base environment
    """

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)
        super().__init__(env)

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return np.clip(action, self.action_space.low, self.action_space.high)
