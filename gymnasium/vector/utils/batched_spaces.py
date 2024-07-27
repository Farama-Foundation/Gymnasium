"""Batching support for Spaces of same type but possibly varying low/high values."""

from __future__ import annotations

from copy import deepcopy
from functools import singledispatch

import numpy as np

from gymnasium import Space
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    MultiBinary,
    MultiDiscrete,
    OneOf,
    Sequence,
    Text,
    Tuple,
)


@singledispatch
def batch_differing_spaces(spaces: list[Space]):
    """Batch a Sequence of spaces that allows the subspaces to contain minor differences."""
    assert len(spaces) > 0
    assert all(isinstance(space, type(spaces[0])) for space in spaces)
    assert type(spaces[0]) in batch_differing_spaces.registry

    return batch_differing_spaces.dispatch(type(spaces[0]))(spaces)


@batch_differing_spaces.register(Box)
def _batch_differing_spaces_box(spaces: list[Box]):
    assert all(spaces[0].dtype == space for space in spaces)

    return Box(
        low=np.array([space.low for space in spaces]),
        high=np.array([space.high for space in spaces]),
        dtype=spaces[0].dtype,
        seed=deepcopy(spaces[0].np_random),
    )


@batch_differing_spaces.register(Discrete)
def _batch_differing_spaces_discrete(spaces: list[Discrete]):
    return MultiDiscrete(
        nvec=np.array([space.n for space in spaces]),
        start=np.array([space.start for space in spaces]),
        seed=deepcopy(spaces[0].np_random),
    )


@batch_differing_spaces.register(MultiDiscrete)
def _batch_differing_spaces_multi_discrete(spaces: list[MultiDiscrete]):
    return Box(
        low=np.array([space.start for space in spaces]),
        high=np.array([space.start + space.nvec for space in spaces]) - 1,
        dtype=spaces[0].dtype,
        seed=deepcopy(spaces[0].np_random),
    )


@batch_differing_spaces.register(MultiBinary)
def _batch_differing_spaces_multi_binary(spaces: list[MultiBinary]):
    assert all(spaces[0].shape == space.shape for space in spaces)

    return Box(
        low=0,
        high=1,
        shape=(len(spaces),) + spaces[0].shape,
        dtype=spaces[0].dtype,
        seed=deepcopy(spaces[0].np_random),
    )


@batch_differing_spaces.register(Tuple)
def _batch_differing_spaces_tuple(spaces: list[Tuple]):
    return Tuple(
        tuple(
            batch_differing_spaces(subspaces)
            for subspaces in zip(*[space.spaces for space in spaces])
        ),
        seed=deepcopy(spaces[0].np_random),
    )


@batch_differing_spaces.register(Dict)
def _batch_differing_spaces_dict(spaces: list[Dict]):
    assert all(spaces[0].keys() == space.keys() for space in spaces)

    return Dict(
        {
            key: batch_differing_spaces([space[key] for space in spaces])
            for key in spaces[0].keys()
        },
        seed=deepcopy(spaces[0].np_random),
    )


@batch_differing_spaces.register(Graph)
@batch_differing_spaces.register(Text)
@batch_differing_spaces.register(Sequence)
@batch_differing_spaces.register(OneOf)
def _batch_spaces_undefined(spaces: list[Graph | Text | Sequence | OneOf]):
    return Tuple(spaces, seed=deepcopy(spaces[0].np_random))


def all_spaces_have_same_shape(spaces):
    """Check if all spaces have the same size."""
    if not spaces:
        return True  # An empty list is considered to have the same shape

    def get_space_shape(space):
        if isinstance(space, Box):
            return space.shape
        elif isinstance(space, Discrete):
            return ()  # Discrete spaces are considered scalar
        elif isinstance(space, Dict):
            return tuple(get_space_shape(s) for s in space.spaces.values())
        elif isinstance(space, Tuple):
            return tuple(get_space_shape(s) for s in space.spaces)
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    first_shape = get_space_shape(spaces[0])
    return all(get_space_shape(space) == first_shape for space in spaces[1:])


def all_spaces_have_same_type(spaces):
    """Check if all spaces have the same space type (Box, Discrete, etc)."""
    if not spaces:
        return True  # An empty list is considered to have the same type

    # Get the type of the first space
    first_type = type(spaces[0])

    # Check if all spaces have the same type as the first one
    return all(isinstance(space, first_type) for space in spaces)
