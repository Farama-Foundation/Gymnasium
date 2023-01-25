"""Numpy utility functions: concatenate space samples and create empty array."""
from __future__ import annotations

from collections import OrderedDict
from functools import singledispatch
from typing import Any, Callable, Iterable

import numpy as np

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


__all__ = ["concatenate", "create_empty_array"]


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
def create_empty_array(
    space: Space, n: int = 1, fn: Callable[..., np.ndarray] = np.zeros
) -> tuple[Any, ...] | dict[str, Any] | np.ndarray:
    """Create an empty (possibly nested) (normally numpy-based) array, used in conjunction with ``concatenate(..., out=array)``.

    In most cases, the array will be contained within the batched space, however, this is not guaranteed.

    Example::

        >>> from gymnasium.spaces import Box, Dict
        >>> import numpy as np
        >>> space = Dict({
        ... 'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
        ... 'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)})
        >>> create_empty_array(space, n=2, fn=np.zeros)
        OrderedDict([('position', array([[0., 0., 0.],
               [0., 0., 0.]], dtype=float32)), ('velocity', array([[0., 0.],
               [0., 0.]], dtype=float32))])

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
    if isinstance(space, Space):
        raise ValueError(
            f"Space of type `{type(space)}` doesn't have an registered `create_empty_array` function."
        )
    else:
        raise TypeError(
            f"The space provided to `create_empty_array` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@create_empty_array.register(Box)
@create_empty_array.register(Discrete)
@create_empty_array.register(MultiDiscrete)
@create_empty_array.register(MultiBinary)
def _create_empty_array_multi(space: Box, n: int = 1, fn=np.zeros) -> np.ndarray:
    shape = space.shape if n is None else (n,) + space.shape
    return fn(shape, dtype=space.dtype)


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
