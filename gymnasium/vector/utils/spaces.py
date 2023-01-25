"""Utility functions for gymnasium spaces: batch space and iterator."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Any, Iterable, Iterator

import numpy as np

from gymnasium.error import CustomSpaceError
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Space,
    Text,
    Tuple,
)


BaseGymSpaces = (Box, Discrete, MultiDiscrete, MultiBinary)
_BaseGymSpaces = BaseGymSpaces
__all__ = ["BaseGymSpaces", "_BaseGymSpaces", "batch_space", "iterate"]


@singledispatch
def batch_space(space: Space, n: int = 1) -> Space:
    """Create a (batched) space, containing multiple copies of a single space.

    Example::

        >>> from gymnasium.spaces import Box, Dict
        >>> import numpy as np
        >>> space = Dict({
        ...     'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
        ...     'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)
        ... })
        >>> batch_space(space, n=5)
        Dict('position': Box(0.0, 1.0, (5, 3), float32), 'velocity': Box(0.0, 1.0, (5, 2), float32))

    Args:
        space: Space (e.g. the observation space) for a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment.

    Returns:
        Space (e.g. the observation space) for a batch of environments in the vectorized environment.

    Raises:
        ValueError: Cannot batch space does not have a registered function.
    """
    if isinstance(space, Space):
        raise ValueError(
            f"Space of type `{type(space)}` doesn't have an registered `batch_space` function."
        )
    else:
        raise TypeError(
            f"The space provided to `batch_space` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@batch_space.register(Box)
def _batch_space_box(space: Box, n: int = 1):
    repeats = tuple([n] + [1] * space.low.ndim)
    low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
    return Box(low=low, high=high, dtype=space.dtype, seed=deepcopy(space.np_random))


@batch_space.register(Discrete)
def _batch_space_discrete(space: Discrete, n: int = 1):
    if space.start == 0:
        return MultiDiscrete(
            np.full((n,), space.n, dtype=space.dtype),
            dtype=space.dtype,
            seed=deepcopy(space.np_random),
        )
    else:
        return Box(
            low=space.start,
            high=space.start + space.n - 1,
            shape=(n,),
            dtype=space.dtype,
            seed=deepcopy(space.np_random),
        )


@batch_space.register(MultiDiscrete)
def _batch_space_multidiscrete(space: MultiDiscrete, n: int = 1):
    repeats = tuple([n] + [1] * space.nvec.ndim)
    high = np.tile(space.nvec, repeats) - 1
    return Box(
        low=np.zeros_like(high),
        high=high,
        dtype=space.dtype,
        seed=deepcopy(space.np_random),
    )


@batch_space.register(MultiBinary)
def _batch_space_multibinary(space: MultiBinary, n: int = 1):
    return Box(
        low=0,
        high=1,
        shape=(n,) + space.shape,
        dtype=space.dtype,
        seed=deepcopy(space.np_random),
    )


@batch_space.register(Tuple)
def _batch_space_tuple(space: Tuple, n: int = 1):
    return Tuple(
        tuple(batch_space(subspace, n=n) for subspace in space.spaces),
        seed=deepcopy(space.np_random),
    )


@batch_space.register(Dict)
def _batch_space_dict(space: Dict, n: int = 1):
    return Dict(
        {key: batch_space(subspace, n=n) for key, subspace in space.items()},
        seed=deepcopy(space.np_random),
    )


@batch_space.register(Graph)
@batch_space.register(Text)
@batch_space.register(Sequence)
def _batch_space_custom(space: Tuple, n: int = 1):
    # Without deepcopy, then the space.np_random is batched_space.spaces[0].np_random
    # Which is an issue if you are sampling actions of both the original space and the batched space
    batched_space = Tuple(
        tuple(deepcopy(space) for _ in range(n)), seed=deepcopy(space.np_random)
    )
    new_seeds = list(map(int, space.np_random.integers(0, 1_000_000, n)))
    batched_space.seed(new_seeds)
    return batched_space


@singledispatch
def iterate(space: Space, items: Iterable) -> Iterator:
    """Iterate over the elements of a (batched) space.

    Example::

        >>> from gymnasium.spaces import Box, Dict
        >>> import numpy as np
        >>> space = Dict({
        ... 'position': Box(low=0, high=1, shape=(2, 3), seed=42, dtype=np.float32),
        ... 'velocity': Box(low=0, high=1, shape=(2, 2), seed=42, dtype=np.float32)})
        >>> items = space.sample()
        >>> it = iterate(space, items)
        >>> next(it)
        OrderedDict([('position', array([0.77395606, 0.43887845, 0.85859793], dtype=float32)), ('velocity', array([0.77395606, 0.43887845], dtype=float32))])
        >>> next(it)
        OrderedDict([('position', array([0.697368  , 0.09417735, 0.97562236], dtype=float32)), ('velocity', array([0.85859793, 0.697368  ], dtype=float32))])
        >>> next(it) # doctest: +SKIP
        StopIteration

    Args:
        space: Space to which `items` belong to.
        items: Items to be iterated over.

    Returns:
        Iterator over the elements in `items`.

    Raises:
        ValueError: Space is not an instance of :class:`gymnasium.Space`
    """
    if isinstance(space, Space):
        raise ValueError(
            f"Space of type `{type(space)}` doesn't have an registered `iterate` function."
        )
    else:
        raise TypeError(
            f"The space provided to `iterate` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@iterate.register(Discrete)
def _iterate_discrete(space: Discrete, items: Iterable):
    raise TypeError("Unable to iterate over a space of type `Discrete`.")


@iterate.register(Box)
@iterate.register(MultiDiscrete)
@iterate.register(MultiBinary)
def _iterate_base(space: Box | MultiDiscrete | MultiBinary, items: np.ndarray):
    try:
        return iter(items)
    except TypeError as e:
        raise TypeError(
            f"Unable to iterate over the following elements: {items}"
        ) from e


@iterate.register(Tuple)
def _iterate_tuple(space: Tuple, items: tuple[Any, ...]):
    # If this is a tuple of custom subspaces only, then simply iterate over items
    if all(type(subspace) in iterate.registry for subspace in space):
        return zip(*[iterate(subspace, items[i]) for i, subspace in enumerate(space)])

    try:
        return iter(items)
    except Exception as e:
        unregistered_spaces = [
            type(subspace)
            for subspace in space
            if type(subspace) not in iterate.registry
        ]
        raise CustomSpaceError(
            f"Could not iterate through {space} as no custom iterate function is registered for {unregistered_spaces} and `iter(items)` raised the following error: {e}."
        ) from e


@iterate.register(Dict)
def _iterate_dict(space, items):
    keys, values = zip(
        *[
            (key, iterate(subspace, items[key]))
            for key, subspace in space.spaces.items()
        ]
    )
    for item in zip(*values):
        yield OrderedDict({key: value for key, value in zip(keys, item)})
