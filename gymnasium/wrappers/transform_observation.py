"""Wrapper for transforming observations."""
from typing import Any, Callable

import gymnasium as gym


class TransformObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Transform the observation via an arbitrary function :attr:`f`.

    The function :attr:`f` should be defined on the observation space of the base environment, ``env``, and should, ideally, return values in the same space.

    If the transformation you wish to apply to observations returns values in a *different* space, you should subclass :class:`ObservationWrapper`, implement the transformation, and set the new observation space accordingly. If you were to use this wrapper instead, the observation space would be set incorrectly.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformObservation
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformObservation(env, lambda obs: obs + 0.1 * np.random.randn(*obs.shape))
        >>> env.reset(seed=42)
        (array([0.20380084, 0.03390356, 0.13373359, 0.24382612]), {})
    """

    def __init__(self, env: gym.Env, f: Callable[[Any], Any]):
        """Initialize the :class:`TransformObservation` wrapper with an environment and a transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """
        gym.utils.RecordConstructorArgs.__init__(self, f=f)
        gym.ObservationWrapper.__init__(self, env)

        assert callable(f)
        self.f = f

    def observation(self, observation):
        """Transforms the observations with callable :attr:`f`.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """
        return self.f(observation)
