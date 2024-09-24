"""Utility functions for the wrappers."""

from __future__ import annotations

from functools import singledispatch
from typing import Callable

import numpy as np

from gymnasium import Space
from gymnasium.error import CustomSpaceError
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    GraphInstance,
    MultiBinary,
    MultiDiscrete,
    OneOf,
    Sequence,
    Text,
    Tuple,
)
from gymnasium.spaces.space import T_cov


__all__ = ["RunningMeanStd", "update_mean_var_count_from_moments", "create_zero_array"]


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), dtype=np.float64):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


@singledispatch
def create_zero_array(space: Space[T_cov]) -> T_cov:
    """Creates a zero-based array of a space, this is similar to ``create_empty_array`` except all arrays are valid samples from the space.

    As some ``Box`` cases have ``high`` or ``low`` that don't contain zero then the ``create_empty_array`` would in case
    create arrays which is not contained in the space.

    Args:
        space: The space to create a zero array for

    Returns:
        Valid sample from the space that is as close to zero as possible
    """
    if isinstance(space, Space):
        raise CustomSpaceError(
            f"Space of type `{type(space)}` doesn't have an registered `create_zero_array` function. Register `{type(space)}` for `create_zero_array` to support it."
        )
    else:
        raise TypeError(
            f"The space provided to `create_zero_array` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@create_zero_array.register(Box)
def _create_box_zero_array(space: Box):
    zero_array = np.zeros(space.shape, dtype=space.dtype)
    zero_array = np.where(space.low > 0, space.low, zero_array)
    zero_array = np.where(space.high < 0, space.high, zero_array)
    return zero_array


@create_zero_array.register(Discrete)
def _create_discrete_zero_array(space: Discrete):
    return space.start


@create_zero_array.register(MultiDiscrete)
def _create_multidiscrete_zero_array(space: MultiDiscrete):
    return np.array(space.start, copy=True, dtype=space.dtype)


@create_zero_array.register(MultiBinary)
def _create_array_zero_array(space: MultiBinary):
    return np.zeros(space.shape, dtype=space.dtype)


@create_zero_array.register(Tuple)
def _create_tuple_zero_array(space: Tuple):
    return tuple(create_zero_array(subspace) for subspace in space.spaces)


@create_zero_array.register(Dict)
def _create_dict_zero_array(space: Dict):
    return {key: create_zero_array(subspace) for key, subspace in space.spaces.items()}


@create_zero_array.register(Sequence)
def _create_sequence_zero_array(space: Sequence):
    if space.stack:
        return create_zero_array(space.stacked_feature_space)
    else:
        return tuple()


@create_zero_array.register(Text)
def _create_text_zero_array(space: Text):
    return "".join(space.characters[0] for _ in range(space.min_length))


@create_zero_array.register(Graph)
def _create_graph_zero_array(space: Graph):
    nodes = np.expand_dims(create_zero_array(space.node_space), axis=0)
    if space.edge_space is None:
        return GraphInstance(nodes=nodes, edges=None, edge_links=None)
    else:
        edges = np.expand_dims(create_zero_array(space.edge_space), axis=0)
        edge_links = np.zeros((1, 2), dtype=np.int64)
        return GraphInstance(nodes=nodes, edges=edges, edge_links=edge_links)


@create_zero_array.register(OneOf)
def _create_one_of_zero_array(space: OneOf):
    return 0, create_zero_array(space.spaces[0])


def rescale_box(
    box: Box,
    new_min: np.floating | np.integer | np.ndarray,
    new_max: np.floating | np.integer | np.ndarray,
) -> tuple[Box, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Rescale and shift the given box space to match the given bounds.

    For unbounded components in the original space, the corresponding target bounds must also be infinite and vice versa.

    Args:
        box: The box space to rescale
        new_min: The new minimum bound
        new_max: The new maximum bound

    Returns:
        A tuple containing the rescaled box space, the forward transformation function (original -> rescaled) and the
        backward transformation function (rescaled -> original).
    """
    assert isinstance(box, Box)

    if not isinstance(new_min, np.ndarray):
        assert np.issubdtype(type(new_min), np.integer) or np.issubdtype(
            type(new_min), np.floating
        )
        new_min = np.full(box.shape, new_min)
    assert (
        new_min.shape == box.shape
    ), f"{new_min.shape}, {box.shape}, {new_min}, {box.low}"

    if not isinstance(new_max, np.ndarray):
        assert np.issubdtype(type(new_max), np.integer) or np.issubdtype(
            type(new_max), np.floating
        )
        new_max = np.full(box.shape, new_max)
    assert new_max.shape == box.shape
    assert np.all((new_min == box.low)[np.isinf(new_min) | np.isinf(box.low)])
    assert np.all((new_max == box.high)[np.isinf(new_max) | np.isinf(box.high)])
    assert np.all(new_min <= new_max)
    assert np.all(box.low <= box.high)

    # Imagine the x-axis between the old Box and the y-axis being the new Box
    # float128 is not available everywhere
    try:
        high_low_diff_dtype = np.float128
    except AttributeError:
        high_low_diff_dtype = np.float64

    min_finite = np.isfinite(new_min)
    max_finite = np.isfinite(new_max)
    both_finite = min_finite & max_finite

    high_low_diff = np.array(
        box.high[both_finite], dtype=high_low_diff_dtype
    ) - np.array(box.low[both_finite], dtype=high_low_diff_dtype)

    gradient = np.ones_like(new_min, dtype=box.dtype)
    gradient[both_finite] = (
        new_max[both_finite] - new_min[both_finite]
    ) / high_low_diff

    intercept = np.zeros_like(new_min, dtype=box.dtype)
    # In cases where both are finite, the lower operation takes precedence
    intercept[max_finite] = new_max[max_finite] - box.high[max_finite]
    intercept[min_finite] = (
        gradient[min_finite] * -box.low[min_finite] + new_min[min_finite]
    )

    new_box = Box(
        low=new_min,
        high=new_max,
        shape=box.shape,
        dtype=box.dtype,
    )

    def forward(obs: np.ndarray) -> np.ndarray:
        return gradient * obs + intercept

    def backward(obs: np.ndarray) -> np.ndarray:
        return (obs - intercept) / gradient

    return new_box, forward, backward
