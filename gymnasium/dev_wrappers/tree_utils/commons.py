"""A set of utility functions for lambda wrappers."""
from copy import deepcopy
from typing import Any, Callable, Sequence, Union

from gymnasium.dev_wrappers import ArgType, ParameterType, TreeParameterType
from gymnasium.dev_wrappers.tree_utils.transform_space_bounds import (
    transform_space_bounds,
)
from gymnasium.spaces import Dict, Tuple


@transform_space_bounds.register(Tuple)
def _process_space_tuple(
    space: Tuple,
    args: TreeParameterType,
    func: Callable[[Union[ArgType, ParameterType]], Any],
) -> Tuple:
    assert isinstance(args, Sequence)
    assert len(space) == len(args)

    updated_space = [s for s in space]

    for i, arg in enumerate(args):
        updated_space[i] = func(space[i], arg, func)

    return Tuple(updated_space)


@transform_space_bounds.register(Dict)
def _process_space_dict(
    space: Dict,
    args: TreeParameterType,
    func: Callable[[Union[ArgType, ParameterType]], Any],
) -> Dict:
    assert isinstance(args, dict)
    updated_space = deepcopy(space)

    for arg in args:
        updated_space[arg] = func(space[arg], args[arg], func)

    return updated_space
