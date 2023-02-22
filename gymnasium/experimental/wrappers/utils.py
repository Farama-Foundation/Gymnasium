"""Utility functions for the wrappers."""
from collections import OrderedDict
from functools import singledispatch

import numpy as np

from gymnasium import Space
from gymnasium.core import ObsType
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
    Text,
    Tuple,
)


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
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
def create_zero_obs(space: Space[ObsType]) -> ObsType:
    if isinstance(space, Space):
        raise CustomSpaceError(
            f"Space of type `{type(space)}` doesn't have an registered `create_zero_obs` function. Register `{type(space)}` for `create_zero_obs` to support it."
        )
    else:
        raise TypeError(
            f"The space provided to `create_zero_obs` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@create_zero_obs.register(Box)
def _create_box_zero_obs(space: Box):
    zero_array = np.zeros_like(space.shape)
    zero_array = np.where(space.low > 0, space.low, zero_array)
    zero_array = np.where(space.high < 0, space.high, zero_array)
    return zero_array


@create_zero_obs.register(Discrete)
def _create_discrete_zero_obs(space: Discrete):
    return space.start


@create_zero_obs.register(MultiDiscrete)
@create_zero_obs.register(MultiBinary)
def _create_array_zero_obs(space: Discrete):
    return np.zeros_like(space.shape)


@create_zero_obs.register(Tuple)
def _create_tuple_zero_obs(space: Tuple):
    return tuple(create_zero_obs(subspace) for subspace in space.spaces)


@create_zero_obs.register(Dict)
def _create_dict_zero_obs(space: Dict):
    return OrderedDict(
        {key: create_zero_obs(subspace) for key, subspace in space.spaces.items()}
    )


@create_zero_obs.register(Sequence)
def _create_sequence_zero_obs(space: Sequence):
    if space.stack:
        return create_zero_obs(space.batched_feature_space)
    else:
        return tuple()


@create_zero_obs.register(Text)
def _create_text_zero_obs(space: Text):
    return "".join(space.characters[0] for _ in range(space.min_length))


@create_zero_obs.register(Graph)
def _create_graph_zero_obs(space: Graph):
    if space.edge_space is None:
        return GraphInstance(nodes=create_zero_obs(space.node_space))
    else:
        # TODO
        return GraphInstance()
