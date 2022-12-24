"""Space-based utility functions for vector environments.

- ``batch_space``: Create a (batched) space, containing multiple copies of a single space.
- ``concatenate``: Concatenate multiple samples from (unbatched) space into a single object.
- ``Iterate``: Iterate over the elements of a (batched) space and items.
- ``create_empty_array``: Create an empty (possibly nested) (normally numpy-based) array, used in conjunction with ``concatenate(..., out=array)``
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Any, Iterable, Iterator

import numpy as np

from gymnasium.error import CustomSpaceError
from gymnasium.logger import warn
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    GraphInstance,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Space,
    Text,
    Tuple,
)


@singledispatch
def batch_space(space: Space, n: int = 1) -> Space:
    """Create a (batched) space, containing multiple copies of a single space.

    Example::

        >>> from gymnasium.spaces import Box, Dict
        >>> space = Dict({
        ...     'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
        ...     'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)
        ... })
        >>> batch_space(space, n=5)
        Dict(position:Box(5, 3), velocity:Box(5, 2))

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


@batch_space.register(Space)
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
def concatenate(
    space: Space, items: Iterable, out: tuple[Any, ...] | dict[str, Any] | np.ndarray
) -> tuple[Any, ...] | dict[str, Any] | np.ndarray:
    """Concatenate multiple samples from space into a single object.

    Example::

        >>> from gymnasium.spaces import Box
        >>> space = Box(low=0, high=1, shape=(3,), dtype=np.float32)
        >>> out = np.zeros((2, 3), dtype=np.float32)
        >>> items = [space.sample() for _ in range(2)]
        >>> concatenate(space, items, out)
        array([[0.6348213 , 0.28607962, 0.60760117],
               [0.87383074, 0.192658  , 0.2148103 ]], dtype=float32)

    Args:
        space: Observation space of a single environment in the vectorized environment.
        items: Samples to be concatenated.
        out: The output object. This object is a (possibly nested) numpy array.

    Returns:
        The output object. This object is a (possibly nested) numpy array.

    Raises:
        ValueError: Space
    """
    if isinstance(space, Space):
        raise ValueError(
            f"Space of type `{type(space)}` doesn't have an registered `concatenate` function."
        )
    else:
        raise TypeError(
            f"The space provided to `concatenate` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@concatenate.register(Box)
@concatenate.register(Discrete)
@concatenate.register(MultiDiscrete)
@concatenate.register(MultiBinary)
def _concatenate_base(
    space: Box | Discrete | MultiDiscrete | MultiBinary,
    items: Iterable,
    out: np.ndarray,
) -> np.ndarray:
    return np.stack(items, axis=0, out=out)


@concatenate.register(Tuple)
def _concatenate_tuple(
    space: Tuple, items: Iterable, out: tuple[Any, ...]
) -> tuple[Any, ...]:
    return tuple(
        concatenate(subspace, [item[i] for item in items], out[i])
        for (i, subspace) in enumerate(space.spaces)
    )


@concatenate.register(Dict)
def _concatenate_dict(
    space: Dict, items: Iterable, out: dict[str, Any]
) -> dict[str, Any]:
    return OrderedDict(
        {
            key: concatenate(subspace, [item[key] for item in items], out[key])
            for key, subspace in space.items()
        }
    )


@concatenate.register(Space)
@concatenate.register(Graph)
@concatenate.register(Text)
@concatenate.register(Sequence)
def _concatenate_custom(space: Space, items: Iterable, out: None) -> tuple[Any, ...]:
    if out is not None:
        warn(
            f"For {type(space)} concatenate, `out` is not None ({out}) however the value is ignored."
        )
    return tuple(items)


@singledispatch
def iterate(space: Space, items: Iterable) -> Iterator:
    """Iterate over the elements of a (batched) space.

    Example::

        >>> from gymnasium.spaces import Box, Dict
        >>> space = Dict({
        ... 'position': Box(low=0, high=1, shape=(2, 3), dtype=np.float32),
        ... 'velocity': Box(low=0, high=1, shape=(2, 2), dtype=np.float32)})
        >>> items = space.sample()
        >>> it = iterate(space, items)
        >>> next(it)
        {'position': array([-0.99644893, -0.08304597, -0.7238421 ], dtype=float32),
        'velocity': array([0.35848552, 0.1533453 ], dtype=float32)}
        >>> next(it)
        {'position': array([-0.67958736, -0.49076623,  0.38661423], dtype=float32),
        'velocity': array([0.7975036 , 0.93317133], dtype=float32)}
        >>> next(it)
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


@singledispatch
def create_empty_array(
    space: Space, n: int = 1, fn: callable = np.zeros
) -> tuple[Any, ...] | dict[str, Any] | np.ndarray:
    """Create an empty (possibly nested) (normally numpy-based) array, used in conjunction with ``concatenate(..., out=array)``.

    Example::

        >>> from gymnasium.spaces import Box, Dict
        >>> space = Dict({
        ... 'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
        ... 'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)})
        >>> create_empty_array(space, n=2, fn=np.zeros)
        OrderedDict([('position', array([[0., 0., 0.],
                                         [0., 0., 0.]], dtype=float32)),
                     ('velocity', array([[0., 0.],
                                         [0., 0.]], dtype=float32))])

    Args:
        space: Observation space of a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment. If `None`, creates an empty sample from `space`.
        fn: Function to apply when creating the empty numpy array. Examples of such functions are `np.empty` or `np.zeros`.

    Returns:
        The output object. This object is a (possibly nested) numpy array.

    Raises:
        ValueError: Space is not a valid :class:`gym.Space` instance
    """
    if isinstance(space, Space):
        raise ValueError(
            f"Space of type `{type(space)}` doesn't have an registered `create_empty_array` function."
        )
    else:
        raise TypeError(
            f"The space provided to `create_empty_array` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@create_empty_array.register(Box)
def _create_empty_array_box(space: Box, n: int = 1, fn=np.zeros) -> np.ndarray:
    return fn((n,) + space.shape, dtype=space.dtype)


@create_empty_array.register(Discrete)
def _create_empty_array_discrete(
    space: Discrete, n: int = 1, fn=np.zeros
) -> np.ndarray:
    return fn((n,), dtype=space.dtype)


@create_empty_array.register(MultiDiscrete)
@create_empty_array.register(MultiBinary)
def _create_empty_array_multi(
    space: MultiDiscrete | MultiBinary, n: int = 1, fn=np.zeros
) -> np.ndarray:
    return fn((n,) + space.shape, dtype=space.dtype)


@create_empty_array.register(Tuple)
def _create_empty_array_tuple(space: Tuple, n: int = 1, fn=np.zeros) -> tuple[Any, ...]:
    return tuple(create_empty_array(subspace, n=n, fn=fn) for subspace in space.spaces)


@create_empty_array.register(Dict)
def _create_empty_array_dict(space: Dict, n: int = 1, fn=np.zeros) -> dict[str, Any]:
    return OrderedDict(
        {
            key: create_empty_array(subspace, n=n, fn=fn)
            for key, subspace in space.items()
        }
    )


@create_empty_array.register(Graph)
def _create_empty_array_graph(
    space: Graph, n: int = 1, fn=np.zeros
) -> tuple[GraphInstance, ...]:
    if space.edge_space is not None:
        return tuple(
            GraphInstance(
                nodes=fn((1,) + space.node_space.shape, dtype=space.node_space.dtype),
                edges=fn((1,) + space.edge_space.shape, dtype=space.edge_space.dtype),
                edge_links=fn((1, 2), dtype=np.int64),
            )
            for _ in range(n)
        )
    else:
        return tuple(
            GraphInstance(
                nodes=fn((1,) + space.node_space.shape, dtype=space.node_space.dtype),
                edges=None,
                edge_links=None,
            )
            for _ in range(n)
        )


@create_empty_array.register(Text)
def _create_empty_array_text(space: Text, n: int = 1, fn=np.zeros) -> tuple[str, ...]:
    return tuple(space.characters[0] * space.min_length for _ in range(n))


@create_empty_array.register(Sequence)
def _create_empty_array_sequence(
    space: Sequence, n: int = 1, fn=np.zeros
) -> tuple[tuple[()], ...]:
    return tuple(tuple() for _ in range(n))
