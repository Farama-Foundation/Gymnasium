"""A set of utility functions for lambda wrappers."""
from copy import deepcopy
from functools import singledispatch
from typing import Any, Callable, Sequence, Union

from gymnasium.dev_wrappers import ArgType, ParameterType, TreeParameterType
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Space,
    Tuple,
)


@singledispatch
def make_scale_args(
    space: Space,
    args: TreeParameterType,
    func: Callable[[Union[ArgType, ParameterType]], Any],
):
    """Compute args for rescaling action function.

    Action space args needs to be extended in order
    to correctly rescale the actions.
    i.e. args before: {"body":{"left_arm": (-0.5,0.5)}, ...}
    args after: {"body":{"left_arm": (-0.5,0.5,-1,1)}, ...}
    where -1, 1 was the old action space bound.
    old action space is needed to rescale actions.
    """
    raise NotImplementedError


@make_scale_args.register(Discrete)
@make_scale_args.register(MultiDiscrete)
@make_scale_args.register(MultiBinary)
def _make_scale_args_not_scalable(
    space: Space,
    args: TreeParameterType,
    func: Callable[[Union[ArgType, ParameterType]], Any],
):
    """Do nothing in case of not scalable spaces.

    Trying to rescale `Discrete`, `Multibinary` and `MultiDiscrete`
    spaces has no effect.
    """


@make_scale_args.register(Box)
def _make_scale_args_box(
    space: Box, args: Sequence, func: Callable[[Union[ArgType, ParameterType]], Any]
) -> ParameterType:
    if args is None:
        return (space.low, space.high, space.low, space.high)
    return (*args, space.low, space.high)


@make_scale_args.register(Dict)
def _make_scale_args_dict(
    space: Dict,
    args: TreeParameterType,
    func: Callable[[Union[ArgType, ParameterType]], Any],
) -> TreeParameterType:
    extended_args = deepcopy(args)

    for arg in args:
        extended_args[arg] = func(space[arg], args[arg], func)

    return extended_args


@make_scale_args.register(Tuple)
def _make_scale_args_tuple(
    space: Tuple,
    args: TreeParameterType,
    func: Callable[[Union[ArgType, ParameterType]], Any],
) -> TreeParameterType:
    assert isinstance(args, Sequence)
    assert len(space) == len(args)

    extended_args = [arg for arg in args]

    for i in range(len(args)):
        extended_args[i] = func(space[i], args[i], func)
    return extended_args
