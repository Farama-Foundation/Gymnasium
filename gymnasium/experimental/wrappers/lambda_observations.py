"""Lambda observation wrappers which apply a function to the observation."""

from typing import Any, Callable

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.experimental.wrappers import ArgType


class LambdaObservationV0(gym.ObservationWrapper):
    """Lambda observation wrapper where a function is provided that is applied to the observation."""

    def __init__(
        self,
        env: gym.Env,
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
