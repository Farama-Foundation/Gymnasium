"""A set of utility functions for lambda wrappers."""
import typing
from functools import singledispatch
from typing import Any, Callable

import jumpy as jp

from gymnasium.dev_wrappers import FuncArgType
from gymnasium.error import InvalidSpaceOperation
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space


@singledispatch
def reshape_space(space: Space, args: FuncArgType, fn: Callable) -> Any:
    """Reshape space with the provided args."""
    raise NotImplementedError


@reshape_space.register(Discrete)
@reshape_space.register(MultiBinary)
@reshape_space.register(MultiDiscrete)
def _reshape_space_not_reshapable(space, args: FuncArgType, fn: Callable):
    """Return original space shape for not reshable space.

    Trying to reshape `Discrete`, `Multibinary` and `MultiDiscrete`
    spaces has no effect.
    """
    if args:
        raise InvalidSpaceOperation(f"Cannot reshape a space of type {type(space)}.")
    return space


@reshape_space.register(Box)
def _reshape_space_box(space, args: FuncArgType[typing.Tuple[int, int]], fn: Callable):
    """Reshape `Box` space."""
    if not args:
        return space
    return Box(
        jp.reshape(space.low, args),
        jp.reshape(space.high, args),
        shape=args,
        dtype=space.dtype,
    )
