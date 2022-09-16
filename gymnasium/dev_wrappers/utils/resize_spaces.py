"""A set of utility functions for lambda wrappers."""
import typing
from functools import singledispatch
from typing import Any, Callable

import numpy as np
import tinyscaler

from gymnasium.dev_wrappers import FuncArgType
from gymnasium.error import InvalidSpaceOperation
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space


@singledispatch
def resize_space(
    space: Space, args: FuncArgType[typing.Tuple[int, int]], fn: Callable
) -> Any:
    """Resize space with the provided args."""
    raise NotImplementedError


@resize_space.register(Discrete)
@resize_space.register(MultiBinary)
@resize_space.register(MultiDiscrete)
def _resize_space_not_reshapable(
    space, args: FuncArgType[typing.Tuple[int, int]], fn: Callable
):
    """Return original space shape for not reshable space.

    Trying to reshape `Discrete`, `Multibinary` and `MultiDiscrete`
    spaces has no effect.
    """
    if args:
        raise InvalidSpaceOperation(f"Cannot resize a space of type {type(space)}.")
    return space


@resize_space.register(Box)
def _resize_space_box(space, args: FuncArgType[typing.Tuple[int, int]], fn: Callable):
    if args is not None:
        if len(space.shape) == 4:  # vectorized environment
            num_envs = space.low.shape[0]
            new_lows = tinyscaler.scale(space.low[0, :], args, mode="bilinear")
            new_highs = tinyscaler.scale(space.low[0, :], args, mode="bilinear")

            space_low = np.repeat(new_lows[None, ...], num_envs, axis=0)
            space_high = np.repeat(new_highs[None, ...], num_envs, axis=0)
        else:
            space_low = tinyscaler.scale(space.low, args, mode="bilinear")
            space_high = tinyscaler.scale(space.high, args, mode="bilinear")

        shape = space_low.shape

        return Box(
            space_low,
            space_high,
            shape=shape,
            dtype=space.dtype,
        )
    return space
