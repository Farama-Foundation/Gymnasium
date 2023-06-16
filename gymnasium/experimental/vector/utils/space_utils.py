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
from gymnasium.spaces.space import T_cov


__all__ = ["batch_space", "iterate", "concatenate", "create_empty_array"]


@singledispatch
def batch_space(space: Space[Any], n: int = 1) -> Space[Any]:
    """Create a (batched) space, containing multiple copies of a single space.

    Args:
        space: Space (e.g. the observation space) for a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment.

    Returns:
        Space (e.g. the observation space) for a batch of environments in the vectorized environment.

    Raises:
        ValueError: Cannot batch space does not have a registered function.

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
    return MultiDiscrete(
        np.full((n,), space.n, dtype=space.dtype),
        dtype=space.dtype,
        seed=deepcopy(space.np_random),
        start=np.full((n,), space.start, dtype=space.dtype),
    )


@batch_space.register(MultiDiscrete)
def _batch_space_multidiscrete(space: MultiDiscrete, n: int = 1):
    repeats = tuple([n] + [1] * space.nvec.ndim)
    low = np.tile(space.start, repeats)
    high = low + np.tile(space.nvec, repeats) - 1
    return Box(
        low=low,
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
@batch_space.register(Space)
def _batch_space_custom(space: Graph | Text | Sequence, n: int = 1):
    # Without deepcopy, then the space.np_random is batched_space.spaces[0].np_random
    # Which is an issue if you are sampling actions of both the original space and the batched space
    batched_space = Tuple(
        tuple(deepcopy(space) for _ in range(n)), seed=deepcopy(space.np_random)
    )
    space_rng = deepcopy(space.np_random)
    new_seeds = list(map(int, space_rng.integers(0, 1e8, n)))
    batched_space.seed(new_seeds)
    return batched_space


@singledispatch
def iterate(space: Space[T_cov], items: Iterable[T_cov]) -> Iterator:
    """Iterate over the elements of a (batched) space.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        items: Samples to be concatenated.
        out: The output object. This object is a (possibly nested) numpy array.

    Returns:
        The output object. This object is a (possibly nested) numpy array.

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
    if isinstance(space, Space):
        raise CustomSpaceError(
            f"Space of type `{type(space)}` doesn't have an registered `iterate` function. Register `{type(space)}` for `iterate` to support it."
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
def _iterate_dict(space: Dict, items: dict[str, Any]):
    keys, values = zip(
        *[
            (key, iterate(subspace, items[key]))
            for key, subspace in space.spaces.items()
        ]
    )
    for item in zip(*values):
        yield OrderedDict({key: value for key, value in zip(keys, item)})


@singledispatch
def concatenate(
    space: Space, items: Iterable, out: tuple[Any, ...] | dict[str, Any] | np.ndarray
) -> tuple[Any, ...] | dict[str, Any] | np.ndarray:
    """Concatenate multiple samples from space into a single object.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        items: Samples to be concatenated.
        out: The output object. This object is a (possibly nested) numpy array.

    Returns:
        The output object. This object is a (possibly nested) numpy array.

    Raises:
        ValueError: Space

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


@concatenate.register(Graph)
@concatenate.register(Text)
@concatenate.register(Sequence)
@concatenate.register(Space)
def _concatenate_custom(space: Space, items: Iterable, out: None) -> tuple[Any, ...]:
    return tuple(items)


@singledispatch
def create_empty_array(
    space: Space, n: int = 1, fn: callable = np.zeros
) -> tuple[Any, ...] | dict[str, Any] | np.ndarray:
    """Create an empty (possibly nested) (normally numpy-based) array, used in conjunction with ``concatenate(..., out=array)``.

    In most cases, the array will be contained within the batched space, however, this is not guaranteed.

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
    raise TypeError(
        f"The space provided to `create_empty_array` is not a gymnasium Space instance, type: {type(space)}, {space}"
    )


# It is possible for the some of the Box low to be greater than 0, then array is not in space
@create_empty_array.register(Box)
# If the Discrete start > 0 or start + length < 0 then array is not in space
@create_empty_array.register(Discrete)
@create_empty_array.register(MultiDiscrete)
@create_empty_array.register(MultiBinary)
def _create_empty_array_multi(space: Box, n: int = 1, fn=np.zeros) -> np.ndarray:
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
) -> tuple[Any, ...]:
    if space.stack:
        return tuple(
            create_empty_array(space.feature_space, n=1, fn=fn) for _ in range(n)
        )
    else:
        return tuple(tuple() for _ in range(n))


@create_empty_array.register(Space)
def _create_empty_array_custom(space, n=1, fn=np.zeros):
    return None
