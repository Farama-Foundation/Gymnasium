"""Utility functions for gymnasium spaces: `batch_space` and `iterator`."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Callable, Iterable, Iterator

import numpy as np

from gymnasium.error import CustomSpaceError
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Space,
    Tuple,
)


__all__ = ["batch_space", "iterate", "concatenate", "create_empty_array"]


@singledispatch
def batch_space(space: Space, n: int = 1) -> Space:
    """Create a (batched) space, containing multiple copies of a single space.

    Args:
        space: Space (e.g. the observation space) for a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment.

    Returns:
        Space (e.g. the observation space) for a batch of environments in the vectorized environment.

    Raises:
        ValueError: Cannot batch space that is not a valid :class:`gym.Space` instance

    Example:
        >>> from gymnasium.spaces import Box, Dict
        >>> import numpy as np
        >>> space = Dict({
        ...     'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
        ...     'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)
        ... })
        >>> batch_space(space, n=5)
        Dict('position': Box(0.0, 1.0, (5, 3), float32), 'velocity': Box(0.0, 1.0, (5, 2), float32))
    """
    raise ValueError(
        f"Cannot batch space with type `{type(space)}`. The space must be a valid `gymnasium.Space` instance."
    )


@batch_space.register(Box)
def _batch_space_box(space, n=1):
    repeats = tuple([n] + [1] * space.low.ndim)
    low, high = np.tile(space.low, repeats), np.tile(space.high, repeats)
    return Box(low=low, high=high, dtype=space.dtype, seed=deepcopy(space.np_random))


@batch_space.register(Discrete)
def _batch_space_discrete(space, n=1):
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
def _batch_space_multidiscrete(space, n=1):
    repeats = tuple([n] + [1] * space.nvec.ndim)
    high = np.tile(space.nvec, repeats) - 1
    return Box(
        low=np.zeros_like(high),
        high=high,
        dtype=space.dtype,
        seed=deepcopy(space.np_random),
    )


@batch_space.register(MultiBinary)
def _batch_space_multibinary(space, n=1):
    return Box(
        low=0,
        high=1,
        shape=(n,) + space.shape,
        dtype=space.dtype,
        seed=deepcopy(space.np_random),
    )


@batch_space.register(Tuple)
def _batch_space_tuple(space, n=1):
    return Tuple(
        tuple(batch_space(subspace, n=n) for subspace in space.spaces),
        seed=deepcopy(space.np_random),
    )


@batch_space.register(Dict)
def _batch_space_dict(space, n=1):
    return Dict(
        OrderedDict(
            [
                (key, batch_space(subspace, n=n))
                for (key, subspace) in space.spaces.items()
            ]
        ),
        seed=deepcopy(space.np_random),
    )


@batch_space.register(Space)
def _batch_space_custom(space, n=1):
    # Without deepcopy, then the space.np_random is batched_space.spaces[0].np_random
    # Which is an issue if you are sampling actions of both the original space and the batched space
    batched_space = Tuple(
        tuple(deepcopy(space) for _ in range(n)), seed=deepcopy(space.np_random)
    )
    new_seeds = list(map(int, batched_space.np_random.integers(0, 1e8, n)))
    batched_space.seed(new_seeds)
    return batched_space


@singledispatch
def iterate(space: Space, items) -> Iterator:
    """Iterate over the elements of a (batched) space.

    Args:
        space: Space to which `items` belong to.
        items: Items to be iterated over.

    Returns:
        Iterator over the elements in `items`.

    Raises:
        ValueError: Space is not an instance of :class:`gym.Space`

    Example:
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
        >>> next(it)
        Traceback (most recent call last):
            ...
        StopIteration
    """
    raise ValueError(
        f"Space of type `{type(space)}` is not a valid `gymnasium.Space` instance."
    )


@iterate.register(Discrete)
def _iterate_discrete(space, items):
    raise TypeError("Unable to iterate over a space of type `Discrete`.")


@iterate.register(Box)
@iterate.register(MultiDiscrete)
@iterate.register(MultiBinary)
def _iterate_base(space, items):
    try:
        return iter(items)
    except TypeError as e:
        raise TypeError(
            f"Unable to iterate over the following elements: {items}"
        ) from e


@iterate.register(Tuple)
def _iterate_tuple(space, items):
    # If this is a tuple of custom subspaces only, then simply iterate over items
    if all(
        isinstance(subspace, Space)
        and (not isinstance(subspace, (Box, Discrete, MultiDiscrete, Tuple, Dict)))
        for subspace in space.spaces
    ):
        return iter(items)

    return zip(
        *[iterate(subspace, items[i]) for i, subspace in enumerate(space.spaces)]
    )


@iterate.register(Dict)
def _iterate_dict(space, items):
    keys, values = zip(
        *[
            (key, iterate(subspace, items[key]))
            for key, subspace in space.spaces.items()
        ]
    )
    for item in zip(*values):
        yield OrderedDict([(key, value) for (key, value) in zip(keys, item)])


@iterate.register(Space)
def _iterate_custom(space, items):
    raise CustomSpaceError(
        f"Unable to iterate over {items}, since {space} "
        "is a custom `gymnasium.Space` instance (i.e. not one of "
        "`Box`, `Dict`, etc...)."
    )


@singledispatch
def concatenate(
    space: Space, items: Iterable, out: tuple | dict | np.ndarray
) -> tuple | dict | np.ndarray:
    """Concatenate multiple samples from space into a single object.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        items: Samples to be concatenated.
        out: The output object. This object is a (possibly nested) numpy array.

    Returns:
        The output object. This object is a (possibly nested) numpy array.

    Raises:
        ValueError: Space is not a valid :class:`gym.Space` instance

    Example:
        >>> from gymnasium.spaces import Box
        >>> import numpy as np
        >>> space = Box(low=0, high=1, shape=(3,), seed=42, dtype=np.float32)
        >>> out = np.zeros((2, 3), dtype=np.float32)
        >>> items = [space.sample() for _ in range(2)]
        >>> concatenate(space, items, out)
        array([[0.77395606, 0.43887845, 0.85859793],
               [0.697368  , 0.09417735, 0.97562236]], dtype=float32)
    """
    raise ValueError(
        f"Space of type `{type(space)}` is not a valid `gymnasium.Space` instance."
    )


@concatenate.register(Box)
@concatenate.register(Discrete)
@concatenate.register(MultiDiscrete)
@concatenate.register(MultiBinary)
def _concatenate_base(space, items, out):
    return np.stack(items, axis=0, out=out)


@concatenate.register(Tuple)
def _concatenate_tuple(space, items, out):
    return tuple(
        concatenate(subspace, [item[i] for item in items], out[i])
        for (i, subspace) in enumerate(space.spaces)
    )


@concatenate.register(Dict)
def _concatenate_dict(space, items, out):
    return OrderedDict(
        [
            (key, concatenate(subspace, [item[key] for item in items], out[key]))
            for (key, subspace) in space.spaces.items()
        ]
    )


@concatenate.register(Space)
def _concatenate_custom(space, items, out):
    return tuple(items)


@singledispatch
def create_empty_array(
    space: Space, n: int = 1, fn: Callable[..., np.ndarray] = np.zeros
) -> tuple | dict | np.ndarray:
    """Create an empty (possibly nested) numpy array.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment. If `None`, creates an empty sample from `space`.
        fn: Function to apply when creating the empty numpy array. Examples of such functions are `np.empty` or `np.zeros`.

    Returns:
        The output object. This object is a (possibly nested) numpy array.

    Raises:
        ValueError: Space is not a valid :class:`gym.Space` instance

    Example:
        >>> from gymnasium.spaces import Box, Dict
        >>> import numpy as np
        >>> space = Dict({
        ... 'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
        ... 'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)})
        >>> create_empty_array(space, n=2, fn=np.zeros)
        OrderedDict([('position', array([[0., 0., 0.],
               [0., 0., 0.]], dtype=float32)), ('velocity', array([[0., 0.],
               [0., 0.]], dtype=float32))])
    """
    raise ValueError(
        f"Space of type `{type(space)}` is not a valid `gymnasium.Space` instance."
    )


@create_empty_array.register(Box)
@create_empty_array.register(Discrete)
@create_empty_array.register(MultiDiscrete)
@create_empty_array.register(MultiBinary)
def _create_empty_array_base(space, n=1, fn=np.zeros):
    shape = space.shape if (n is None) else (n,) + space.shape
    return fn(shape, dtype=space.dtype)


@create_empty_array.register(Tuple)
def _create_empty_array_tuple(space, n=1, fn=np.zeros):
    return tuple(create_empty_array(subspace, n=n, fn=fn) for subspace in space.spaces)


@create_empty_array.register(Dict)
def _create_empty_array_dict(space, n=1, fn=np.zeros):
    return OrderedDict(
        [
            (key, create_empty_array(subspace, n=n, fn=fn))
            for (key, subspace) in space.spaces.items()
        ]
    )


@create_empty_array.register(Space)
def _create_empty_array_custom(space, n=1, fn=np.zeros):
    return None
