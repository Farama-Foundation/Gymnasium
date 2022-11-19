"""Lambda action wrapper which apply a function to the provided action."""

from typing import Any, Callable

import gymnasium as gym
from gymnasium.core import ActType
from gymnasium.dev_wrappers import ArgType


class LambdaActionV0(gym.ActionWrapper):
    """A wrapper that provides a function to modify the action passed to :meth:`step`."""

    def __init__(
        self,
        env: gym.Env,
        func: Callable[[ArgType], Any],
    ):
        """Initialize LambdaAction.

        Args:
            env (Env): The gymnasium environment
            func (Callable): function to apply to action
        """
        super().__init__(env)

        self.func = func

    def action(self, action: ActType) -> Any:
        """Apply function to action."""
        return self.func(action)
