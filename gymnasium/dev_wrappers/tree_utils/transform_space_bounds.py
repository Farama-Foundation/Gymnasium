"""A set of utility functions for lambda wrappers."""
from functools import singledispatch
from typing import Any, Callable, Union

from gymnasium.dev_wrappers import ArgType, ParameterType, TreeParameterType
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space


@singledispatch
def transform_space_bounds(
    space: Space,
    args: TreeParameterType,
    func: Callable[[Union[ArgType, ParameterType]], Any],
):
    """Transform space bounds with the provided args."""
    raise NotImplementedError


@transform_space_bounds.register(Discrete)
@transform_space_bounds.register(MultiBinary)
@transform_space_bounds.register(MultiDiscrete)
def _transform_space_discrete(
    space: Union[Discrete, MultiBinary, MultiDiscrete],
    args: TreeParameterType,
    func: Callable[[Union[ArgType, ParameterType]], Any],
) -> Union[Discrete, MultiBinary, MultiDiscrete]:
    return space


@transform_space_bounds.register(Box)
def _transform_space_box(
    space: Box,
    args: TreeParameterType,
    func: Callable[[Union[ArgType, ParameterType]], Any],
) -> Box:
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
