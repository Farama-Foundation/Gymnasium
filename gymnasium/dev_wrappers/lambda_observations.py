"""Lambda observation wrappers which apply a function to the observation."""

from typing import Any, Callable

import gymnasium
from gymnasium.core import ObsType
from gymnasium.dev_wrappers import ArgType


class LambdaObservationsV0(gymnasium.ObservationWrapper):
    """Lambda observation wrapper where a function is provided that is applied to the observation.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.spaces import Dict, Discrete
        >>> from gymnasium.wrappers import LambdaObservationsV0
        >>> env = gym.make("CartPole-v1")
        >>> env = LambdaObservationsV0(env, lambda obs: obs * 100)
        >>> _ = env.reset()
        >>> obs, rew, term, trunc, info = env.step(1)
        >>> obs
        array([ 9.995892, 432.83587, 23.945259, -626.16], dtype=float32)
    """

    def __init__(
        self,
        env: gymnasium.Env,
        func: Callable[[ArgType], Any],
    ):
        """Constructor for the lambda observation wrapper.

        Args:
            env: The environment to wrap
            func: A function that takes
        """
        super().__init__(env)

        self.func = func

    def observation(self, observation: ObsType) -> Any:
        """Apply function to the observation."""
        return self.func(observation)
