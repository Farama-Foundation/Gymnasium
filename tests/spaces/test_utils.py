from itertools import zip_longest

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Box, Graph, Sequence, utils
from gymnasium.spaces.utils import is_space_dtype_shape_equiv
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector.utils import (
    batch_space,
    create_shared_memory,
    iterate,
    read_from_shared_memory,
    write_to_shared_memory,
)
from tests.spaces.utils import TESTING_SPACES, TESTING_SPACES_IDS


TESTING_SPACES_EXPECTED_FLATDIMS = [
    # Discrete
    3,
    3,
    4,
    # Box
    1,
    4,
    2,
    2,
    2,
    12,
    3,
    4,
    # Multi-discrete
    4,
    10,
    4,
    10,
    5,
    5,
    # Multi-binary
    8,
    6,
    # Text
    6,
    6,
    6,
    # Tuple
    9,
    7,
    10,
    6,
    None,
    # Dict
    7,
    8,
    17,
    None,
    # Graph
    None,
    None,
    None,
    None,
    # Sequence
    None,
    None,
    None,
    None,
    None,
    # OneOf
    4,
    5,
]
assert len(TESTING_SPACES) == len(TESTING_SPACES_EXPECTED_FLATDIMS)


@pytest.mark.parametrize(
    ["space", "flatdim"],
    zip_longest(TESTING_SPACES, TESTING_SPACES_EXPECTED_FLATDIMS),
    ids=TESTING_SPACES_IDS,
)
def test_flatdim(space: gym.spaces.Space, flatdim: int | None):
    """Checks that the flattened dims of the space is equal to an expected value."""
    if space.is_np_flattenable:
        dim = utils.flatdim(space)
        assert dim == flatdim, f"Expected {dim} to equal {flatdim}"
    else:
        with pytest.raises(
            ValueError,
        ):
            utils.flatdim(space)


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_flatten_space(space):
    """Test that the flattened spaces are a box and have the `flatdim` shape."""
    flat_space = utils.flatten_space(space)

    if space.is_np_flattenable:
        assert isinstance(flat_space, Box)
        (single_dim,) = flat_space.shape
        flatdim = utils.flatdim(space)

        assert single_dim == flatdim
    elif isinstance(flat_space, Graph):
        assert isinstance(space, Graph)

        (node_single_dim,) = flat_space.node_space.shape
        node_flatdim = utils.flatdim(space.node_space)
        assert node_single_dim == node_flatdim

        if flat_space.edge_space is not None:
            (edge_single_dim,) = flat_space.edge_space.shape
            edge_flatdim = utils.flatdim(space.edge_space)
            assert edge_single_dim == edge_flatdim
    else:
        assert isinstance(
            space,
            (gym.spaces.Tuple, gym.spaces.Dict, gym.spaces.Sequence),
        )


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_flatten(space):
    """Test that a flattened sample have the `flatdim` shape."""
    sample = space.sample()
    flattened_sample = utils.flatten(space, sample)

    if space.is_np_flattenable:
        assert isinstance(flattened_sample, np.ndarray)
        (single_dim,) = flattened_sample.shape
        flatdim = utils.flatdim(space)

        assert single_dim == flatdim
    else:
        assert isinstance(space, Sequence) or isinstance(
            flattened_sample, (tuple, dict, Graph)
        )


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_flat_space_contains_flat_points(space):
    """Test that the flattened samples are contained within the flattened space."""
    flattened_samples = [utils.flatten(space, space.sample()) for _ in range(10)]
    flat_space = utils.flatten_space(space)

    for flat_sample in flattened_samples:
        assert flat_sample in flat_space


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_flatten_roundtripping(space):
    """Tests roundtripping with flattening and unflattening are equal to the original sample."""
    samples = [space.sample() for _ in range(10)]

    flattened_samples = [utils.flatten(space, sample) for sample in samples]
    unflattened_samples = [
        utils.unflatten(space, sample) for sample in flattened_samples
    ]

    for original, roundtripped in zip(samples, unflattened_samples):
        assert data_equivalence(original, roundtripped)


@pytest.mark.parametrize(
    ["space", "flattened_sample"],
    [
        (gym.spaces.Discrete(3), np.array([0, 1, 0])),
        (gym.spaces.Discrete(3, dtype=np.int16), np.array([0, 1, 0], dtype=np.int16)),
        (gym.spaces.Discrete(3, dtype=np.int32), np.array([0, 1, 0], dtype=np.int32)),
    ],
)
def test_unflatten_discrete_with_dtype(space, flattened_sample):
    unflattened = utils.unflatten(space, flattened_sample)
    assert unflattened == 1
    assert space.dtype == unflattened.dtype


def test_unflatten_discrete_error():
    value = np.array([0])
    with pytest.raises(ValueError):
        utils.unflatten(gym.spaces.Discrete(1), value)


def test_unflatten_multidiscrete_error():
    value = np.array([0, 0])
    with pytest.raises(ValueError):
        utils.unflatten(gym.spaces.MultiDiscrete([1, 1]), value)


@pytest.mark.parametrize("space", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_is_space_dtype_shape_equiv(space):
    assert is_space_dtype_shape_equiv(space, space) is True


@pytest.mark.parametrize("space_1", TESTING_SPACES, ids=TESTING_SPACES_IDS)
def test_all_space_pairs_for_is_space_dtype_shape_equiv(space_1):
    """Practically check that the `is_space_dtype_shape_equiv` works as expected for `shared_memory`."""
    for space_2 in TESTING_SPACES:
        compatible = is_space_dtype_shape_equiv(space_1, space_2)

        if compatible:
            try:
                shared_memory = create_shared_memory(space_1, n=2)
            except TypeError as err:
                assert (
                    "has a dynamic shape so its not possible to make a static shared memory."
                    in str(err)
                )
                continue

            batched_space = batch_space(space_1, n=2)

            space_1.seed(123)
            space_2.seed(123)
            sample_1 = space_1.sample()
            sample_2 = space_2.sample()

            write_to_shared_memory(space_1, 0, sample_1, shared_memory)
            write_to_shared_memory(space_2, 1, sample_2, shared_memory)

            read_samples = read_from_shared_memory(space_1, shared_memory, n=2)
            read_sample_1, read_sample_2 = iterate(batched_space, read_samples)

            assert data_equivalence(sample_1, read_sample_1)
            assert data_equivalence(sample_2, read_sample_2)
