"""A set of utility functions for lambda wrappers."""
import typing
from functools import singledispatch
from typing import Any

from gymnasium.dev_wrappers import FuncArgType
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
def filter_space(space: Space, args: FuncArgType) -> Any:
    """Filter space with the provided args."""
    raise NotImplementedError


@filter_space.register(Box)
@filter_space.register(Discrete)
@filter_space.register(MultiBinary)
@filter_space.register(MultiDiscrete)
def _filter_space_box(space: Space, args: FuncArgType):
    return space


@filter_space.register(Dict)
def _filter_space_dict(space: Space, args: FuncArgType[typing.Dict[str, bool]]):
    """Filter `Dict` observation space by args."""
    return Dict(
        [
            (name, filter_space(value, args.get(name)))
            for name, value in space.items()
            if args.get(name, False)
        ]
    )


@filter_space.register(Tuple)
def _filter_space_tuple(space: Space, args: FuncArgType[typing.Tuple[bool]]):
    """Filter `Tuple` observation space by args."""
    return Tuple([filter_space(value, arg) for value, arg in zip(space, args) if arg])
