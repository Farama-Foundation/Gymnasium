"""A set of utility functions for lambda wrappers."""
import typing
from functools import singledispatch
from typing import Any, Callable

from gymnasium.dev_wrappers import FuncArgType
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space


@singledispatch
def transform_space_bounds(
    space: Space, args: FuncArgType[typing.Tuple[int, int]], fn: Callable
) -> Any:
    """Transform space bounds with the provided args."""
    raise NotImplementedError


@transform_space_bounds.register(Discrete)
@transform_space_bounds.register(MultiBinary)
@transform_space_bounds.register(MultiDiscrete)
def _transform_space_discrete(
    space, args: FuncArgType[typing.Tuple[int, int]], fn: Callable
):
    return space


@transform_space_bounds.register(Box)
def _transform_space_box(
    space, args: FuncArgType[typing.Tuple[int, int]], fn: Callable
):
    """Change `Box` space low and high value."""
    if not args:
        return space
    low, high = args

    return Box(
        low if low is not None else space.low,
        high if high is not None else space.high,
        shape=space.shape,
        dtype=space.dtype,
    )
