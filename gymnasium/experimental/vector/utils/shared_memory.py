"""Utility functions for vector environments to share memory between processes."""
from __future__ import annotations

import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Any

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
    flatten,
)


__all__ = ["create_shared_memory", "read_from_shared_memory", "write_to_shared_memory"]


@singledispatch
def create_shared_memory(
    space: Space[Any], n: int = 1, ctx=mp
) -> dict[str, Any] | tuple[Any, ...] | mp.Array:
    """Create a shared memory object, to be shared across processes.

    This eventually contains the observations from the vectorized environment.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment (i.e. the number of processes).
        ctx: The multiprocess module

    Returns:
        shared_memory for the shared object across processes.

    Raises:
        CustomSpaceError: Space is not a valid :class:`gymnasium.Space` instance
    """
    if isinstance(space, Space):
        raise CustomSpaceError(
            f"Space of type `{type(space)}` doesn't have an registered `create_shared_memory` function. Register `{type(space)}` for `create_shared_memory` to support it."
        )
    else:
        raise TypeError(
            f"The space provided to `create_shared_memory` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@create_shared_memory.register(Box)
@create_shared_memory.register(Discrete)
@create_shared_memory.register(MultiDiscrete)
@create_shared_memory.register(MultiBinary)
def _create_base_shared_memory(
    space: Box | Discrete | MultiDiscrete | MultiBinary, n: int = 1, ctx=mp
):
    assert space.dtype is not None
    dtype = space.dtype.char
    if dtype in "?":
        dtype = c_bool
    return ctx.Array(dtype, n * int(np.prod(space.shape)))


@create_shared_memory.register(Tuple)
def _create_tuple_shared_memory(space: Tuple, n: int = 1, ctx=mp):
    return tuple(
        create_shared_memory(subspace, n=n, ctx=ctx) for subspace in space.spaces
    )


@create_shared_memory.register(Dict)
def _create_dict_shared_memory(space: Dict, n: int = 1, ctx=mp):
    return OrderedDict(
        [
            (key, create_shared_memory(subspace, n=n, ctx=ctx))
            for (key, subspace) in space.spaces.items()
        ]
    )


@create_shared_memory.register(Text)
def _create_text_shared_memory(space: Text, n: int = 1, ctx=mp):
    return ctx.Array(np.dtype(np.int32).char, n * space.max_length)


@create_shared_memory.register(Graph)
@create_shared_memory.register(Sequence)
def _create_dynamic_shared_memory(space: Graph | Sequence, n: int = 1, ctx=mp):
    raise TypeError(
        f"As {space} has a dynamic shape then it is not possible to make a static shared memory."
    )


@singledispatch
def read_from_shared_memory(
    space: Space, shared_memory: dict | tuple | mp.Array, n: int = 1
) -> dict[str, Any] | tuple[Any, ...] | np.ndarray:
    """Read the batch of observations from shared memory as a numpy array.

    ..notes::
        The numpy array objects returned by `read_from_shared_memory` shares the
        memory of `shared_memory`. Any changes to `shared_memory` are forwarded
        to `observations`, and vice-versa. To avoid any side-effect, use `np.copy`.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        shared_memory: Shared object across processes. This contains the observations from the vectorized environment.
            This object is created with `create_shared_memory`.
        n: Number of environments in the vectorized environment (i.e. the number of processes).

    Returns:
        Batch of observations as a (possibly nested) numpy array.

    Raises:
        CustomSpaceError: Space is not a valid :class:`gymnasium.Space` instance
    """
    if isinstance(space, Space):
        raise CustomSpaceError(
            f"Space of type `{type(space)}` doesn't have an registered `read_from_shared_memory` function. Register `{type(space)}` for `read_from_shared_memory` to support it."
        )
    else:
        raise TypeError(
            f"The space provided to `read_from_shared_memory` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@read_from_shared_memory.register(Box)
@read_from_shared_memory.register(Discrete)
@read_from_shared_memory.register(MultiDiscrete)
@read_from_shared_memory.register(MultiBinary)
def _read_base_from_shared_memory(
    space: Box | Discrete | MultiDiscrete | MultiBinary, shared_memory, n: int = 1
):
    return np.frombuffer(shared_memory.get_obj(), dtype=space.dtype).reshape(
        (n,) + space.shape
    )


@read_from_shared_memory.register(Tuple)
def _read_tuple_from_shared_memory(space: Tuple, shared_memory, n: int = 1):
    return tuple(
        read_from_shared_memory(subspace, memory, n=n)
        for (memory, subspace) in zip(shared_memory, space.spaces)
    )


@read_from_shared_memory.register(Dict)
def _read_dict_from_shared_memory(space: Dict, shared_memory, n: int = 1):
    return OrderedDict(
        [
            (key, read_from_shared_memory(subspace, shared_memory[key], n=n))
            for (key, subspace) in space.spaces.items()
        ]
    )


@read_from_shared_memory.register(Text)
def _read_text_from_shared_memory(space: Text, shared_memory, n: int = 1) -> tuple[str]:
    data = np.frombuffer(shared_memory.get_obj(), dtype=np.int32).reshape(
        (n, space.max_length)
    )

    return tuple(
        "".join(
            [
                space.character_list[val]
                for val in values
                if val < len(space.character_set)
            ]
        )
        for values in data
    )


@singledispatch
def write_to_shared_memory(
    space: Space,
    index: int,
    value: np.ndarray,
    shared_memory: dict[str, Any] | tuple[Any, ...] | mp.Array,
):
    """Write the observation of a single environment into shared memory.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        index: Index of the environment (must be in `[0, num_envs)`).
        value: Observation of the single environment to write to shared memory.
        shared_memory: Shared object across processes. This contains the observations from the vectorized environment.
            This object is created with `create_shared_memory`.

    Raises:
        CustomSpaceError: Space is not a valid :class:`gymnasium.Space` instance
    """
    if isinstance(space, Space):
        raise CustomSpaceError(
            f"Space of type `{type(space)}` doesn't have an registered `write_to_shared_memory` function. Register `{type(space)}` for `write_to_shared_memory` to support it."
        )
    else:
        raise TypeError(
            f"The space provided to `write_to_shared_memory` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@write_to_shared_memory.register(Box)
@write_to_shared_memory.register(Discrete)
@write_to_shared_memory.register(MultiDiscrete)
@write_to_shared_memory.register(MultiBinary)
def _write_base_to_shared_memory(
    space: Box | Discrete | MultiDiscrete | MultiBinary,
    index: int,
    value,
    shared_memory,
):
    size = int(np.prod(space.shape))
    destination = np.frombuffer(shared_memory.get_obj(), dtype=space.dtype)
    np.copyto(
        destination[index * size : (index + 1) * size],
        np.asarray(value, dtype=space.dtype).flatten(),
    )


@write_to_shared_memory.register(Tuple)
def _write_tuple_to_shared_memory(
    space: Tuple, index: int, values: tuple[Any, ...], shared_memory
):
    for value, memory, subspace in zip(values, shared_memory, space.spaces):
        write_to_shared_memory(subspace, index, value, memory)


@write_to_shared_memory.register(Dict)
def _write_dict_to_shared_memory(
    space: Dict, index: int, values: dict[str, Any], shared_memory
):
    for key, subspace in space.spaces.items():
        write_to_shared_memory(subspace, index, values[key], shared_memory[key])


@write_to_shared_memory.register(Text)
def _write_text_to_shared_memory(space: Text, index: int, values: str, shared_memory):
    size = space.max_length
    destination = np.frombuffer(shared_memory.get_obj(), dtype=np.int32)
    np.copyto(
        destination[index * size : (index + 1) * size],
        flatten(space, values),
    )
