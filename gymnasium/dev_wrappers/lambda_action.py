"""Lambda action wrapper which apply a function to the provided action."""

from functools import partial
from typing import Any, Callable, Sequence

import jumpy as jp

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


class ClipActionV0(LambdaActionV0):
    """A wrapper that clips actions passed to :meth:`step` with an upper and lower bound.

    Basic Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ClipActionV0
        >>> env = gym.make("BipedalWalker-v3")
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env = ClipActionV0(env, (-0.5, 0.5))
        >>> env.action_space
        Box(-0.5, 0.5, (4,), float32)

    Clip with only a lower or upper bound:
        >>> env = gym.make('CarRacing-v2')
        >>> env.action_space
        Box([-1.  0.  0.], 1.0, (3,), float32)
        >>> env = ClipActionV0(env, (None, 0.5))
        >>> env.action_space
        Box([-1.  0.  0.], 0.5, (3,), float32)
    """

    def __init__(self, env: gym.Env, args: Sequence):
        """Constructor for the clip action wrapper.

        Args:
            env (Env): The environment to wrap
            args (Sequence): The arguments for clipping the action space
        """
        super().__init__(
            env, partial(lambda action, args: jp.clip(action, *args), args=args)
        )


class RescaleActionsV0(LambdaActionV0):
    """A wrapper that scales actions passed to :meth:`step` with a scale factor.

    Basic Example:
        >>> import gymnasium
        >>> from gymnasium.wrappers import RescaleActionsV0
        >>> env = gym.make('BipedalWalker-v3')
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env = RescaleActionsV0(env, (-0.5, 0.5))
        >>> env.action_space
        Box(-0.5, 0.5, (4,), float32)

    Composite action space example:
        >>> env = ExampleEnv(
        ...    action_space=Dict(left_arm=Box(-2, 2, (1,)), right_arm=Box(-2, 2, (1,))
        ... )
        >>> env = RescaleActionsV0(env, {"left_arm": (-1,1), "right_arm": (-1,1)})
        >>> env.action_space
        Dict(left_arm: Box(-1, 1, (1,), float32), right_arm: Box(-1, 1, (1,), float32))
    """

    def __init__(self, env: gym.Env, args: Sequence):
        """Constructor for the scale action wrapper.

        Args:
            env (Env): The environment to wrap
            args (Sequence): The arguments for scaling the actions
        """
        super().__init__(
            env, partial(lambda action, args: self._scale(action, args), args=args)
        )

    def _scale(self, action: ActType, args: Sequence) -> jp.ndarray:
        new_low, new_high = args
        old_low, old_high = self.action_space.low, self.action_space.high

        return jp.clip(
            old_low
            + (old_high - old_low) * ((action - new_low) / (new_high - new_low)),
            old_low,
            old_high,
        )
