"""Lambda action wrappers that uses jumpy for compatibility with jax (i.e. brax) and numpy environments."""

from typing import Any, Callable

import gymnasium
from gymnasium.core import ActType
from gymnasium.dev_wrappers import ArgType


class LambdaActionV0(gymnasium.ActionWrapper):
    """A wrapper that provides a function to modify the action passed to :meth:`step`.

    Example to convert continuous actions to discrete:
        >>> import gymnasium
        >>> import numpy as np
        >>> from gymnasium.spaces import Dict
        >>> from gymnasium.wrappers import LambdaActionV0
        >>> env = gymnasium.make("CarRacing-v2", continuous=False)
        >>> env = LambdaActionV0(env, lambda action, _: action.astype(np.int32), None)
        >>> env.action_space
        Discrete(5)
        >>> _ = env.reset()
        >>> obs, rew, term, trunc, info = env.step(np.float64(1.2))
    """

    def __init__(
        self,
        env: gymnasium.Env,
        func: Callable[[ArgType], Any],
    ):
        """Initialize LambdaAction.

        Args:
            env (Env): The gymnasium environment
            func (Callable): function to apply to action
            args: function arcuments
            action_space: wrapped environment action space
        """
        super().__init__(env)

        self.func = func

    def action(self, action: ActType) -> Any:
        """Apply function to action."""
        return self.func(action)
