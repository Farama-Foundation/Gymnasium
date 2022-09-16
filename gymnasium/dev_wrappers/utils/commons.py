"""A set of utility functions for lambda wrappers."""
import typing
from copy import deepcopy
from typing import Any, Callable, Sequence

from gymnasium.dev_wrappers import FuncArgType
from gymnasium.dev_wrappers.utils.grayscale_space import grayscale_space
from gymnasium.dev_wrappers.utils.reshape_space import reshape_space
from gymnasium.dev_wrappers.utils.resize_spaces import resize_space
from gymnasium.dev_wrappers.utils.transform_space_bounds import transform_space_bounds
from gymnasium.dev_wrappers.utils.update_dtype import update_dtype
from gymnasium.spaces import Dict, Tuple


@grayscale_space.register(Tuple)
@resize_space.register(Tuple)
@reshape_space.register(Tuple)
@transform_space_bounds.register(Tuple)
@update_dtype.register(Tuple)
def _process_space_tuple(
    space: Tuple, args: FuncArgType[typing.Tuple[Any, ...]], fn: Callable
):
    assert isinstance(args, Sequence)
    assert len(space) == len(args)

    updated_space = [s for s in space]

    for i, arg in enumerate(args):
        updated_space[i] = fn(space[i], arg, fn)

    return Tuple(updated_space)


@grayscale_space.register(Dict)
@resize_space.register(Dict)
@reshape_space.register(Dict)
@transform_space_bounds.register(Dict)
@update_dtype.register(Dict)
def _process_space_dict(
    space: Dict, args: FuncArgType[typing.Dict[str, Any]], fn: Callable
):
    assert isinstance(args, dict)
    updated_space = deepcopy(space)

    for arg in args:
        updated_space[arg] = fn(space[arg], args[arg], fn)

    return updated_space
