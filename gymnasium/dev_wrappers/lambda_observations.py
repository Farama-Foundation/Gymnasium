"""Lambda observation wrappers that uses jumpy for compatibility with jax (i.e. brax) and numpy environments."""

from typing import Any, Callable

import gymnasium
from gymnasium.core import ObsType
from gymnasium.dev_wrappers import ArgType


class LambdaObservationsV0(gymnasium.ObservationWrapper):
    """Lambda observation wrapper where a function is provided that is applied to the observation.

    Example:
        >>> import gymnasium
        >>> from gymnasium.spaces import Dict, Discrete
        >>> from gymnasium.wrappers import LambdaObservationsV0
        >>> env = gymnasium.make("CartPole-v1")
        >>> env = LambdaObservationsV0(env, lambda obs, arg: obs * arg, 10)
        >>> obs, rew, term, trunc, info = env.step(1)
        >>> obs
        array([ 0.09995892, 4.3283587, 0.23945259, -6.1516], dtype=float32)
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
            args: The arguments that the function takes
            observation_space: The updated observation space
        """
        super().__init__(env)

        self.func = func

    def observation(self, observation: ObsType) -> Any:
        """Apply function to the observation."""
        return self.func(observation)
