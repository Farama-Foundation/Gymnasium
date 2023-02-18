"""Testing `gymnasium.experimental.vector.utils.space_utils` functions."""

import copy
from collections import OrderedDict

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gymnasium.experimental.vector.utils import (
    batch_space,
    concatenate,
    create_empty_array,
    iterate,
)
from gymnasium.spaces import Box, Dict, MultiDiscrete, Space, Tuple
from tests.experimental.vector.testing_utils import (
    BaseGymSpaces,
    CustomSpace,
    assert_rng_equal,
    custom_spaces,
    spaces,
)


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
def test_concatenate(space):
    """Tests the `concatenate` functions with list of spaces."""

    def assert_type(lhs, rhs, n):
        # Special case: if rhs is a list of scalars, lhs must be an np.ndarray
        if np.isscalar(rhs[0]):
            assert isinstance(lhs, np.ndarray)
            assert all([np.isscalar(rhs[i]) for i in range(n)])
        else:
            assert all([isinstance(rhs[i], type(lhs)) for i in range(n)])

    def assert_nested_equal(lhs, rhs, n):
        assert isinstance(rhs, list)
        assert (n > 0) and (len(rhs) == n)
        assert_type(lhs, rhs, n)
        if isinstance(lhs, np.ndarray):
            assert lhs.shape[0] == n
            for i in range(n):
                assert np.all(lhs[i] == rhs[i])

        elif isinstance(lhs, tuple):
            for i in range(len(lhs)):
                rhs_T_i = [rhs[j][i] for j in range(n)]
                assert_nested_equal(lhs[i], rhs_T_i, n)

        elif isinstance(lhs, OrderedDict):
            for key in lhs.keys():
                rhs_T_key = [rhs[j][key] for j in range(n)]
                assert_nested_equal(lhs[key], rhs_T_key, n)

        else:
            raise TypeError(f"Got unknown type `{type(lhs)}`.")

    samples = [space.sample() for _ in range(8)]
    array = create_empty_array(space, n=8)
    concatenated = concatenate(space, samples, array)

    assert np.all(concatenated == array)
    assert_nested_equal(array, samples, n=8)


@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
def test_create_empty_array(space, n):
    """Test `create_empty_array` function with list of spaces and different `n` values."""

    def assert_nested_type(arr, space, n):
        if isinstance(space, BaseGymSpaces):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == space.dtype
            assert arr.shape == (n,) + space.shape

        elif isinstance(space, Tuple):
            assert isinstance(arr, tuple)
            assert len(arr) == len(space.spaces)
            for i in range(len(arr)):
                assert_nested_type(arr[i], space.spaces[i], n)

        elif isinstance(space, Dict):
            assert isinstance(arr, OrderedDict)
            assert set(arr.keys()) ^ set(space.spaces.keys()) == set()
            for key in arr.keys():
                assert_nested_type(arr[key], space.spaces[key], n)

        else:
            raise TypeError(f"Got unknown type `{type(arr)}`.")

    array = create_empty_array(space, n=n, fn=np.empty)
    assert_nested_type(array, space, n=n)


@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
def test_create_empty_array_zeros(space, n):
    """Test `create_empty_array` with a list of spaces and different `n`."""

    def assert_nested_type(arr, space, n):
        if isinstance(space, BaseGymSpaces):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == space.dtype
            assert arr.shape == (n,) + space.shape
            assert np.all(arr == 0)

        elif isinstance(space, Tuple):
            assert isinstance(arr, tuple)
            assert len(arr) == len(space.spaces)
            for i in range(len(arr)):
                assert_nested_type(arr[i], space.spaces[i], n)

        elif isinstance(space, Dict):
            assert isinstance(arr, OrderedDict)
            assert set(arr.keys()) ^ set(space.spaces.keys()) == set()
            for key in arr.keys():
                assert_nested_type(arr[key], space.spaces[key], n)

        else:
            raise TypeError(f"Got unknown type `{type(arr)}`.")

    array = create_empty_array(space, n=n, fn=np.zeros)
    assert_nested_type(array, space, n=n)


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
def test_create_empty_array_none_shape_ones(space):
    """Tests `create_empty_array` with ``None`` space."""

    def assert_nested_type(arr, space):
        if isinstance(space, BaseGymSpaces):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == space.dtype
            assert arr.shape == space.shape
            assert np.all(arr == 1)

        elif isinstance(space, Tuple):
            assert isinstance(arr, tuple)
            assert len(arr) == len(space.spaces)
            for i in range(len(arr)):
                assert_nested_type(arr[i], space.spaces[i])

        elif isinstance(space, Dict):
            assert isinstance(arr, OrderedDict)
            assert set(arr.keys()) ^ set(space.spaces.keys()) == set()
            for key in arr.keys():
                assert_nested_type(arr[key], space.spaces[key])

        else:
            raise TypeError(f"Got unknown type `{type(arr)}`.")

    array = create_empty_array(space, n=None, fn=np.ones)
    assert_nested_type(array, space)


expected_batch_spaces_4 = [
    Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
    Box(low=0.0, high=10.0, shape=(4, 1), dtype=np.float64),
    Box(
        low=np.array(
            [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
        high=np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ),
        dtype=np.float64,
    ),
    Box(
        low=np.array(
            [
                [[-1.0, 0.0], [0.0, -1.0]],
                [[-1.0, 0.0], [0.0, -1.0]],
                [[-1.0, 0.0], [0.0, -1]],
                [[-1.0, 0.0], [0.0, -1.0]],
            ]
        ),
        high=np.ones((4, 2, 2)),
        dtype=np.float64,
    ),
    Box(low=0, high=255, shape=(4,), dtype=np.uint8),
    Box(low=0, high=255, shape=(4, 32, 32, 3), dtype=np.uint8),
    MultiDiscrete([2, 2, 2, 2]),
    Box(low=-2, high=2, shape=(4,), dtype=np.int64),
    Tuple((MultiDiscrete([3, 3, 3, 3]), MultiDiscrete([5, 5, 5, 5]))),
    Tuple(
        (
            MultiDiscrete([7, 7, 7, 7]),
            Box(
                low=np.array([[0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.0, -1]]),
                high=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
                dtype=np.float64,
            ),
        )
    ),
    Box(
        low=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        high=np.array([[10, 12, 16], [10, 12, 16], [10, 12, 16], [10, 12, 16]]),
        dtype=np.int64,
    ),
    Box(low=0, high=1, shape=(4, 19), dtype=np.int8),
    Dict(
        {
            "position": MultiDiscrete([23, 23, 23, 23]),
            "velocity": Box(low=0.0, high=1.0, shape=(4, 1), dtype=np.float64),
        }
    ),
    Dict(
        {
            "position": Dict(
                {
                    "x": MultiDiscrete([29, 29, 29, 29]),
                    "y": MultiDiscrete([31, 31, 31, 31]),
                }
            ),
            "velocity": Tuple(
                (
                    MultiDiscrete([37, 37, 37, 37]),
                    Box(low=0, high=255, shape=(4,), dtype=np.uint8),
                )
            ),
        }
    ),
]

expected_custom_batch_spaces_4 = [
    Tuple((CustomSpace(), CustomSpace(), CustomSpace(), CustomSpace())),
    Tuple(
        (
            Tuple((CustomSpace(), CustomSpace(), CustomSpace(), CustomSpace())),
            Box(low=0, high=255, shape=(4,), dtype=np.uint8),
        )
    ),
]


@pytest.mark.parametrize(
    "space,expected_batch_space_4",
    list(zip(spaces, expected_batch_spaces_4)),
    ids=[space.__class__.__name__ for space in spaces],
)
def test_batch_space(space, expected_batch_space_4):
    """Tests `batch_space` with the expected spaces."""
    batch_space_4 = batch_space(space, n=4)
    assert batch_space_4 == expected_batch_space_4


@pytest.mark.parametrize(
    "space,expected_batch_space_4",
    list(zip(custom_spaces, expected_custom_batch_spaces_4)),
    ids=[space.__class__.__name__ for space in custom_spaces],
)
def test_batch_space_custom_space(space, expected_batch_space_4):
    """Tests `batch_space` for custom spaces with the expected batch spaces."""
    batch_space_4 = batch_space(space, n=4)
    assert batch_space_4 == expected_batch_space_4


@pytest.mark.parametrize(
    "space,batched_space",
    list(zip(spaces, expected_batch_spaces_4)),
    ids=[space.__class__.__name__ for space in spaces],
)
def test_iterate(space, batched_space):
    """Test `iterate` function with list of spaces and expected batch space."""
    items = batched_space.sample()
    iterator = iterate(batched_space, items)
    i = 0
    for i, item in enumerate(iterator):
        assert item in space
    assert i == 3


@pytest.mark.parametrize(
    "space,batched_space",
    list(zip(custom_spaces, expected_custom_batch_spaces_4)),
    ids=[space.__class__.__name__ for space in custom_spaces],
)
def test_iterate_custom_space(space, batched_space):
    """Test iterating over a custom space."""
    items = batched_space.sample()
    iterator = iterate(batched_space, items)
    i = 0
    for i, item in enumerate(iterator):
        assert item in space
    assert i == 3


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
@pytest.mark.parametrize("n", [4, 5], ids=[f"n={n}" for n in [4, 5]])
@pytest.mark.parametrize(
    "base_seed", [123, 456], ids=[f"seed={base_seed}" for base_seed in [123, 456]]
)
def test_rng_different_at_each_index(space: Space, n: int, base_seed: int):
    """Tests that the rng values produced at each index are different to prevent if the rng is copied for each subspace."""
    space.seed(base_seed)

    batched_space = batch_space(space, n)
    assert space.np_random is not batched_space.np_random
    assert_rng_equal(space.np_random, batched_space.np_random)

    batched_sample = batched_space.sample()
    sample = list(iterate(batched_space, batched_sample))
    assert not all(np.all(element == sample[0]) for element in sample), sample


@pytest.mark.parametrize(
    "space", spaces, ids=[space.__class__.__name__ for space in spaces]
)
@pytest.mark.parametrize("n", [1, 2, 5], ids=[f"n={n}" for n in [1, 2, 5]])
@pytest.mark.parametrize(
    "base_seed", [123, 456], ids=[f"seed={base_seed}" for base_seed in [123, 456]]
)
def test_deterministic(space: Space, n: int, base_seed: int):
    """Tests the batched spaces are deterministic by using a copied version."""
    # Copy the spaces and check that the np_random are not reference equal
    space_a = space
    space_a.seed(base_seed)
    space_b = copy.deepcopy(space_a)
    assert_rng_equal(space_a.np_random, space_b.np_random)
    assert space_a.np_random is not space_b.np_random

    # Batch the spaces and check that the np_random are not reference equal
    space_a_batched = batch_space(space_a, n)
    space_b_batched = batch_space(space_b, n)
    assert_rng_equal(space_a_batched.np_random, space_b_batched.np_random)
    assert space_a_batched.np_random is not space_b_batched.np_random
    # Create that the batched space is not reference equal to the origin spaces
    assert space_a.np_random is not space_a_batched.np_random

    # Check that batched space a and b random number generator are not effected by the original space
    space_a.sample()
    space_a_batched_sample = space_a_batched.sample()
    space_b_batched_sample = space_b_batched.sample()
    for a_sample, b_sample in zip(
        iterate(space_a_batched, space_a_batched_sample),
        iterate(space_b_batched, space_b_batched_sample),
    ):
        if isinstance(a_sample, tuple):
            assert len(a_sample) == len(b_sample)
            for a_subsample, b_subsample in zip(a_sample, b_sample):
                assert_array_equal(a_subsample, b_subsample)
        else:
            assert_array_equal(a_sample, b_sample)
