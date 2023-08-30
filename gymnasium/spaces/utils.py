"""Implementation of utility functions that can be applied to spaces.

These functions mostly take care of flattening and unflattening elements of spaces
 to facilitate their usage in learning code.
"""
from __future__ import annotations

import operator as op
import typing
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Any, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
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
def flatdim(space: Space[Any]) -> int:
    """Return the number of dimensions a flattened equivalent of this space would have.

    Args:
        space: The space to return the number of dimensions of the flattened spaces

    Returns:
        The number of dimensions for the flattened spaces

    Raises:
         NotImplementedError: if the space is not defined in :mod:`gym.spaces`.
         ValueError: if the space cannot be flattened into a :class:`gymnasium.spaces.Box`

    Example:
        >>> from gymnasium.spaces import Dict, Discrete
        >>> space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
        >>> flatdim(space)
        5
    """
    if space.is_np_flattenable is False:
        raise ValueError(
            f"{space} cannot be flattened to a numpy array, probably because it contains a `Graph` or `Sequence` subspace"
        )

    raise NotImplementedError(f"Unknown space: `{space}`")


@flatdim.register(Box)
@flatdim.register(MultiBinary)
def _flatdim_box_multibinary(space: Box | MultiBinary) -> int:
    return reduce(op.mul, space.shape, 1)


@flatdim.register(Discrete)
def _flatdim_discrete(space: Discrete) -> int:
    return int(space.n)


@flatdim.register(MultiDiscrete)
def _flatdim_multidiscrete(space: MultiDiscrete) -> int:
    return int(np.sum(space.nvec))


@flatdim.register(Tuple)
def _flatdim_tuple(space: Tuple) -> int:
    if space.is_np_flattenable:
        return sum(flatdim(s) for s in space.spaces)
    raise ValueError(
        f"{space} cannot be flattened to a numpy array, probably because it contains a `Graph` or `Sequence` subspace"
    )


@flatdim.register(Dict)
def _flatdim_dict(space: Dict) -> int:
    if space.is_np_flattenable:
        return sum(flatdim(s) for s in space.spaces.values())
    raise ValueError(
        f"{space} cannot be flattened to a numpy array, probably because it contains a `Graph` or `Sequence` subspace"
    )


@flatdim.register(Graph)
def _flatdim_graph(space: Graph):
    raise ValueError(
        "Cannot get flattened size as the Graph Space in Gym has a dynamic size."
    )


@flatdim.register(Text)
def _flatdim_text(space: Text) -> int:
    return space.max_length


T = TypeVar("T")
FlatType = Union[
    NDArray[Any], typing.Dict[str, Any], typing.Tuple[Any, ...], GraphInstance
]


@singledispatch
def flatten(space: Space[T], x: T) -> FlatType:
    """Flatten a data point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Args:
        space: The space that ``x`` is flattened by
        x: The value to flatten

    Returns:
        The flattened datapoint

            - For :class:`gymnasium.spaces.Box` and :class:`gymnasium.spaces.MultiBinary`, this is a flattened array
            - For :class:`gymnasium.spaces.Discrete` and :class:`gymnasium.spaces.MultiDiscrete`, this is a flattened one-hot array of the sample
            - For :class:`gymnasium.spaces.Tuple` and :class:`gymnasium.spaces.Dict`, this is a concatenated array the subspaces (does not support graph subspaces)
            - For graph spaces, returns :class:`GraphInstance` where:
                - :attr:`GraphInstance.nodes` are n x k arrays
                - :attr:`GraphInstance.edges` are either:
                    - m x k arrays
                    - None
                - :attr:`GraphInstance.edge_links` are either:
                    - m x 2 arrays
                    - None

    Raises:
        NotImplementedError: If the space is not defined in :mod:`gymnasium.spaces`.

    Example:
        >>> from gymnasium.spaces import Box, Discrete, Tuple
        >>> space = Box(0, 1, shape=(3, 5))
        >>> flatten(space, space.sample()).shape
        (15,)
        >>> space = Discrete(4)
        >>> flatten(space, 2)
        array([0, 0, 1, 0])
        >>> space = Tuple((Box(0, 1, shape=(2,)), Box(0, 1, shape=(3,)), Discrete(3)))
        >>> example = ((.5, .25), (1., 0., .2), 1)
        >>> flatten(space, example)
        array([0.5 , 0.25, 1.  , 0.  , 0.2 , 0.  , 1.  , 0.  ])
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@flatten.register(Box)
@flatten.register(MultiBinary)
def _flatten_box_multibinary(space: Box | MultiBinary, x: NDArray[Any]) -> NDArray[Any]:
    return np.asarray(x, dtype=space.dtype).flatten()


@flatten.register(Discrete)
def _flatten_discrete(space: Discrete, x: np.int64) -> NDArray[np.int64]:
    onehot = np.zeros(space.n, dtype=space.dtype)
    onehot[x - space.start] = 1
    return onehot


@flatten.register(MultiDiscrete)
def _flatten_multidiscrete(
    space: MultiDiscrete, x: NDArray[np.int64]
) -> NDArray[np.int64]:
    offsets = np.zeros((space.nvec.size + 1,), dtype=np.int32)
    offsets[1:] = np.cumsum(space.nvec.flatten())

    onehot = np.zeros((offsets[-1],), dtype=space.dtype)
    onehot[offsets[:-1] + (x - space.start).flatten()] = 1
    return onehot


@flatten.register(Tuple)
def _flatten_tuple(space: Tuple, x: tuple[Any, ...]) -> tuple[Any, ...] | NDArray[Any]:
    if space.is_np_flattenable:
        return np.concatenate(
            [np.array(flatten(s, x_part)) for x_part, s in zip(x, space.spaces)]
        )
    return tuple(flatten(s, x_part) for x_part, s in zip(x, space.spaces))


@flatten.register(Dict)
def _flatten_dict(space: Dict, x: dict[str, Any]) -> dict[str, Any] | NDArray[Any]:
    if space.is_np_flattenable:
        return np.concatenate(
            [np.array(flatten(s, x[key])) for key, s in space.spaces.items()]
        )
    return OrderedDict((key, flatten(s, x[key])) for key, s in space.spaces.items())


@flatten.register(Graph)
def _flatten_graph(space: Graph, x: GraphInstance) -> GraphInstance:
    """We're not using ``.unflatten()`` for :class:`Box` and :class:`Discrete` because a graph is not a homogeneous space, see `.flatten` docstring."""

    def _graph_unflatten(
        unflatten_space: Discrete | Box | None,
        unflatten_x: NDArray[Any] | None,
    ) -> NDArray[Any] | None:
        ret = None
        if unflatten_space is not None and unflatten_x is not None:
            if isinstance(unflatten_space, Box):
                ret = unflatten_x.reshape(unflatten_x.shape[0], -1)
            else:
                assert isinstance(unflatten_space, Discrete)
                ret = np.zeros(
                    (unflatten_x.shape[0], unflatten_space.n - unflatten_space.start),
                    dtype=unflatten_space.dtype,
                )
                ret[
                    np.arange(unflatten_x.shape[0]), unflatten_x - unflatten_space.start
                ] = 1
        return ret

    nodes = _graph_unflatten(space.node_space, x.nodes)
    assert nodes is not None
    edges = _graph_unflatten(space.edge_space, x.edges)

    return GraphInstance(nodes, edges, x.edge_links)


@flatten.register(Text)
def _flatten_text(space: Text, x: str) -> NDArray[np.int32]:
    arr = np.full(
        shape=(space.max_length,), fill_value=len(space.character_set), dtype=np.int32
    )
    for i, val in enumerate(x):
        arr[i] = space.character_index(val)
    return arr


@flatten.register(Sequence)
def _flatten_sequence(
    space: Sequence, x: tuple[Any, ...] | Any
) -> tuple[Any, ...] | Any:
    if space.stack:
        samples_iters = gym.vector.utils.iterate(space.stacked_feature_space, x)
        flattened_samples = [
            flatten(space.feature_space, sample) for sample in samples_iters
        ]
        flattened_space = flatten_space(space.feature_space)
        out = gym.vector.utils.create_empty_array(
            flattened_space, n=len(flattened_samples)
        )
        return gym.vector.utils.concatenate(flattened_space, flattened_samples, out)
    else:
        return tuple(flatten(space.feature_space, item) for item in x)


@singledispatch
def unflatten(space: Space[T], x: FlatType) -> T:
    """Unflatten a data point from a space.

    This reverses the transformation applied by :func:`flatten`. You must ensure
    that the ``space`` argument is the same as for the :func:`flatten` call.

    Args:
        space: The space used to unflatten ``x``
        x: The array to unflatten

    Returns:
        A point with a structure that matches the space.

    Raises:
        NotImplementedError: if the space is not defined in :mod:`gymnasium.spaces`.
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@unflatten.register(Box)
@unflatten.register(MultiBinary)
def _unflatten_box_multibinary(
    space: Box | MultiBinary, x: NDArray[Any]
) -> NDArray[Any]:
    return np.asarray(x, dtype=space.dtype).reshape(space.shape)


@unflatten.register(Discrete)
def _unflatten_discrete(space: Discrete, x: NDArray[np.int64]) -> np.int64:
    nonzero = np.nonzero(x)
    if len(nonzero[0]) == 0:
        raise ValueError(
            f"{x} is not a valid one-hot encoded vector and can not be unflattened to space {space}. "
            "Not all valid samples in a flattened space can be unflattened."
        )
    return space.start + nonzero[0][0]


@unflatten.register(MultiDiscrete)
def _unflatten_multidiscrete(
    space: MultiDiscrete, x: NDArray[np.integer[Any]]
) -> NDArray[np.integer[Any]]:
    offsets = np.zeros((space.nvec.size + 1,), dtype=space.dtype)
    offsets[1:] = np.cumsum(space.nvec.flatten())
    nonzero = np.nonzero(x)
    if len(nonzero[0]) == 0:
        raise ValueError(
            f"{x} is not a concatenation of one-hot encoded vectors and can not be unflattened to space {space}. "
            "Not all valid samples in a flattened space can be unflattened."
        )
    (indices,) = cast(type(offsets[:-1]), nonzero)
    return (
        np.asarray(indices - offsets[:-1], dtype=space.dtype).reshape(space.shape)
        + space.start
    )


@unflatten.register(Tuple)
def _unflatten_tuple(
    space: Tuple, x: NDArray[Any] | tuple[Any, ...]
) -> tuple[Any, ...]:
    if space.is_np_flattenable:
        assert isinstance(
            x, np.ndarray
        ), f"{space} is numpy-flattenable. Thus, you should only unflatten numpy arrays for this space. Got a {type(x)}"
        dims = np.asarray([flatdim(s) for s in space.spaces], dtype=np.int_)
        list_flattened = np.split(x, np.cumsum(dims[:-1]))
        return tuple(
            unflatten(s, flattened)
            for flattened, s in zip(list_flattened, space.spaces)
        )
    assert isinstance(
        x, tuple
    ), f"{space} is not numpy-flattenable. Thus, you should only unflatten tuples for this space. Got a {type(x)}"
    return tuple(unflatten(s, flattened) for flattened, s in zip(x, space.spaces))


@unflatten.register(Dict)
def _unflatten_dict(space: Dict, x: NDArray[Any] | dict[str, Any]) -> dict[str, Any]:
    if space.is_np_flattenable:
        dims = np.asarray([flatdim(s) for s in space.spaces.values()], dtype=np.int_)
        list_flattened = np.split(x, np.cumsum(dims[:-1]))
        return OrderedDict(
            [
                (key, unflatten(s, flattened))
                for flattened, (key, s) in zip(list_flattened, space.spaces.items())
            ]
        )
    assert isinstance(
        x, dict
    ), f"{space} is not numpy-flattenable. Thus, you should only unflatten dictionary for this space. Got a {type(x)}"
    return OrderedDict((key, unflatten(s, x[key])) for key, s in space.spaces.items())


@unflatten.register(Graph)
def _unflatten_graph(space: Graph, x: GraphInstance) -> GraphInstance:
    """We're not using `.unflatten() for :class:`Box` and :class:`Discrete` because a graph is not a homogeneous space.

    The size of the outcome is actually not fixed, but determined based on the number of
    nodes and edges in the graph.
    """

    def _graph_unflatten(unflatten_space, unflatten_x):
        result = None
        if unflatten_space is not None and unflatten_x is not None:
            if isinstance(unflatten_space, Box):
                result = unflatten_x.reshape(-1, *unflatten_space.shape)
            elif isinstance(unflatten_space, Discrete):
                result = np.asarray(np.nonzero(unflatten_x))[-1, :]
        return result

    nodes = _graph_unflatten(space.node_space, x.nodes)
    edges = _graph_unflatten(space.edge_space, x.edges)

    return GraphInstance(nodes, edges, x.edge_links)


@unflatten.register(Text)
def _unflatten_text(space: Text, x: NDArray[np.int32]) -> str:
    return "".join(
        [space.character_list[val] for val in x if val < len(space.character_set)]
    )


@unflatten.register(Sequence)
def _unflatten_sequence(space: Sequence, x: tuple[Any, ...]) -> tuple[Any, ...] | Any:
    if space.stack:
        flattened_space = flatten_space(space.feature_space)
        flatten_iters = gym.vector.utils.iterate(flattened_space, x)
        unflattened_samples = [
            unflatten(space.feature_space, sample) for sample in flatten_iters
        ]
        out = gym.vector.utils.create_empty_array(
            space.feature_space, len(unflattened_samples)
        )
        return gym.vector.utils.concatenate(
            space.feature_space, unflattened_samples, out
        )
    else:
        return tuple(unflatten(space.feature_space, item) for item in x)


@singledispatch
def flatten_space(space: Space[Any]) -> Box | Dict | Sequence | Tuple | Graph:
    """Flatten a space into a space that is as flat as possible.

    This function will attempt to flatten ``space`` into a single :class:`gymnasium.spaces.Box` space.
    However, this might not be possible when ``space`` is an instance of :class:`gymnasium.spaces.Graph`,
    :class:`gymnasium.spaces.Sequence` or a compound space that contains a :class:`gymnasium.spaces.Graph`
    or :class:`gymnasium.spaces.Sequence` space.
    This is equivalent to :func:`flatten`, but operates on the space itself. The
    result for non-graph spaces is always a :class:`gymnasium.spaces.Box` with flat boundaries. While
    the result for graph spaces is always a :class:`gymnasium.spaces.Graph` with
    :attr:`Graph.node_space` being a ``Box``
    with flat boundaries and :attr:`Graph.edge_space` being a ``Box`` with flat boundaries or
    ``None``. The box has exactly :func:`flatdim` dimensions. Flattening a sample
    of the original space has the same effect as taking a sample of the flattened
    space. However, sampling from the flattened space is not necessarily reversible.
    For example, sampling from a flattened Discrete space is the same as sampling from
    a Box, and the results may not be integers or one-hot encodings. This may result in
    errors or non-uniform sampling.

    Args:
        space: The space to flatten

    Returns:
        A flattened Box

    Raises:
        NotImplementedError: if the space is not defined in :mod:`gymnasium.spaces`.

    Example:
        Flatten spaces.Box:

        >>> from gymnasium.spaces import Box
        >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(0.0, 1.0, (3, 4, 5), float32)
        >>> flatten_space(box)
        Box(0.0, 1.0, (60,), float32)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

        Flatten spaces.Discrete:

        >>> from gymnasium.spaces import Discrete
        >>> discrete = Discrete(5)
        >>> flatten_space(discrete)
        Box(0, 1, (5,), int64)
        >>> flatten(discrete, discrete.sample()) in flatten_space(discrete)
        True

        Flatten spaces.Dict:

        >>> from gymnasium.spaces import Dict, Discrete, Box
        >>> space = Dict({"position": Discrete(2), "velocity": Box(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(0.0, 1.0, (6,), float64)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True

        Flatten spaces.Graph:

        >>> from gymnasium.spaces import Graph, Discrete, Box
        >>> space = Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5))
        >>> flatten_space(space)
        Graph(Box(-100.0, 100.0, (12,), float32), Box(0, 1, (5,), int64))
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@flatten_space.register(Box)
def _flatten_space_box(space: Box) -> Box:
    return Box(space.low.flatten(), space.high.flatten(), dtype=space.dtype)


@flatten_space.register(Discrete)
@flatten_space.register(MultiBinary)
@flatten_space.register(MultiDiscrete)
def _flatten_space_binary(space: Discrete | MultiBinary | MultiDiscrete) -> Box:
    return Box(low=0, high=1, shape=(flatdim(space),), dtype=space.dtype)


@flatten_space.register(Tuple)
def _flatten_space_tuple(space: Tuple) -> Box | Tuple:
    if space.is_np_flattenable:
        space_list = [flatten_space(s) for s in space.spaces]
        return Box(
            low=np.concatenate([s.low for s in space_list]),
            high=np.concatenate([s.high for s in space_list]),
            dtype=np.result_type(*[s.dtype for s in space_list]),
        )
    return Tuple(spaces=[flatten_space(s) for s in space.spaces])


@flatten_space.register(Dict)
def _flatten_space_dict(space: Dict) -> Box | Dict:
    if space.is_np_flattenable:
        space_list = [flatten_space(s) for s in space.spaces.values()]
        return Box(
            low=np.concatenate([s.low for s in space_list]),
            high=np.concatenate([s.high for s in space_list]),
            dtype=np.result_type(*[s.dtype for s in space_list]),
        )
    return Dict(
        spaces=OrderedDict(
            (key, flatten_space(space)) for key, space in space.spaces.items()
        )
    )


@flatten_space.register(Graph)
def _flatten_space_graph(space: Graph) -> Graph:
    return Graph(
        node_space=flatten_space(space.node_space),
        edge_space=flatten_space(space.edge_space)
        if space.edge_space is not None
        else None,
    )


@flatten_space.register(Text)
def _flatten_space_text(space: Text) -> Box:
    return Box(
        low=0, high=len(space.character_set), shape=(space.max_length,), dtype=np.int32
    )


@flatten_space.register(Sequence)
def _flatten_space_sequence(space: Sequence) -> Sequence:
    return Sequence(flatten_space(space.feature_space), stack=space.stack)
