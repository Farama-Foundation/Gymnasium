"""Lambda action wrapper which apply a function to the provided action."""

from typing import Any, Callable, Optional

import jumpy as jp

import gymnasium as gym
from gymnasium.core import ActType
from gymnasium.dev_wrappers import ArgType, TreeParameterType
from gymnasium.dev_wrappers.tree_utils.make_scale_args import make_scale_args
from gymnasium.dev_wrappers.tree_utils.transform_space_bounds import (
    transform_space_bounds,
)
from gymnasium.spaces.utils import apply_function


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


class LambdaCompositeActionV0(LambdaActionV0):
    """A wrapper that provides a function to modify the action passed to :meth:`step`.

    This wrapper works on composite action action spaces (`Tuple` and `Dict`)
    which supports arbitrarily nested spaces.

    Example:
        >>> env = ExampleEnv(action_space=Dict(left_arm=Discrete(4), right_arm=Box(0.0, 5.0, (1,)))
        >>> env = LambdaActionV0(
        ...     env,
        ...     lambda action, _: action + 10,
        ...     {"right_arm": True},
        ...     None
        ... )
        >>> env.action_space
        Dict(left_arm: Discrete(4), right_arm: Box(0.0, 5.0, (1,), float32))
        >>> _ = env.reset()
        >>> obs, rew, term, trunc, info = env.step({"left_arm": 1, "right_arm": 1})
        >>> info["action"] # the executed action within the environment
        {'action': OrderedDict([('left_arm', 1), ('right_arm', 11)])})
    Vectorized environment:
        >>> env = gymnasium.vector.make("CarRacing-v2", continuous=False, num_envs=2)
        >>> env = LambdaActionV0(
        ...     env, lambda action, _: action.astype(np.int32), [None for _ in range(2)]
        ... )
        >>> obs, rew, term, trunc, info = env.step([np.float64(1.2), np.float64(1.2)])
    """

    def __init__(
        self,
        env: gym.Env,
        func: Callable,
        args: TreeParameterType,
        action_space: Optional[gym.Space] = None,
    ):
        """Initialize LambdaAction.

        Args:
            env (Env): The gymnasium environment
            func (Callable): function to apply to action
            args: function arcuments
            action_space: wrapped environment action space
        """
        super().__init__(env, func)

        self.func_args = args
        if action_space is None:
            self.action_space = env.action_space
        else:
            self.action_space = action_space

    def action(self, action):
        """Apply function to action."""
        return apply_function(self.action_space, action, self.func, self.func_args)

    def _transform_space(self, env: gym.Env, args: TreeParameterType):
        """Process the space and apply the transformation."""
        return transform_space_bounds(env.action_space, args, transform_space_bounds)


class ClipActionsV0(LambdaCompositeActionV0):
    """A wrapper that clips actions passed to :meth:`step` with an upper and lower bound.

    Basic Example:
        >>> import gymnasium
        >>> from gymnasium.wrappers import ClipActionsV0
        >>> env = gymnasium.make("BipedalWalker-v3")
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env = ClipActionsV0(env, (-0.5, 0.5))
        >>> env.action_space
        Box(-0.5, 0.5, (4,), float32)

    Clip with only a lower or upper bound:
        >>> env = gymnasium.make('CarRacing-v1')
        >>> env.action_space
        Box([-1.  0.  0.], 1.0, (3,), float32)
        >>> env = ClipActionsV0(env, (None, 0.5))
        >>> env.action_space
        Box([-1.  0.  0.], 0.5, (3,), float32)

    Composite action space example:
        >>> env = ExampleEnv(action_space=Dict(body=Dict(head=Box(0.0, 10.0, (1,))), left_arm=Discrete(4), right_arm=Box(0.0, 5.0, (1,))))
        >>> env.action_space
        Dict(body: Dict(head: Box(0.0, 10.0, (1,), float32)), left_arm: Discrete(4), right_arm: Box(0.0, 5.0, (1,), float32))
        >>> args = {"right_arm": (0, 2), "body": {"head": (0, 3)}}
        >>> env = ClipActionsV0(env, args)
        >>> env.action_space
        Dict(body: Dict(head: Box(0.0, 3.0, (1,), float32)), left_arm: Discrete(4), right_arm: Box(0.0, 2.0, (1,), float32))
    """

    def __init__(self, env: gym.Env, args: TreeParameterType):
        """Constructor for the clip action wrapper.

        Args:
            env: The environment to wrap
            args: The arguments for clipping the action space
        """
        action_space = self._transform_space(env, args)

        super().__init__(
            env, lambda action, args: jp.clip(action, *args), args, action_space
        )


class ScaleActionsV0(LambdaCompositeActionV0):
    """A wrapper that scales actions passed to :meth:`step` with a scale factor.

    Basic Example:
        >>> import gymnasium
        >>> from gymnasium.wrappers import ScaleActionsV0
        >>> env = gymnasium.make('BipedalWalker-v3')
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env = ScaleActionsV0(env, (-0.5, 0.5))
        >>> env.action_space
        Box(-0.5, 0.5, (4,), float32)

    Composite action space example:
        >>> env = ExampleEnv(
        ...    action_space=Dict(left_arm=Box(-2, 2, (1,)), right_arm=Box(-2, 2, (1,))
        ... )
        >>> env = ScaleActionsV0(env, {"left_arm": (-1,1), "right_arm": (-1,1)})
        >>> env.action_space
        Dict(left_arm: Box(-1, 1, (1,), float32), right_arm: Box(-1, 1, (1,), float32))
    """

    def __init__(self, env: gym.Env, args: TreeParameterType):
        """Constructor for the scale action wrapper.

        Args:
            env: The environment to wrap
            args: The arguments for scaling the actions
        """
        action_space = self._transform_space(env, args)
        args = self._make_scale_args(env, args)

        super().__init__(env, self._scale, args, action_space)

    @staticmethod
    def _scale(action, args):
        new_low, new_high = args[:2]
        old_low, old_high = args[2:]

        return jp.clip(
            old_low
            + (old_high - old_low) * ((action - new_low) / (new_high - new_low)),
            old_low,
            old_high,
        )

    @staticmethod
    def _make_scale_args(env: gym.Env, args: TreeParameterType):
        return make_scale_args(env.action_space, args, make_scale_args)
