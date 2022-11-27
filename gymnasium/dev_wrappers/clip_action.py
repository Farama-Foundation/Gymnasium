"""Wrapper for clipping actions within a valid bound."""
import jumpy as jp

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType


class ClipActionV0(gym.ActionWrapper):
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('Bipedal-Walker-v3')
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env.step(np.array([5.0, 2.0, -10.0, 0.0]))
        # Executes the action np.array([1.0, 1.0, -1.0, 0]) in the base environment
    """

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, spaces.Box)
        super().__init__(env)

    def action(self, action: ActType) -> jp.ndarray:
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return jp.clip(action, self.action_space.low, self.action_space.high)
