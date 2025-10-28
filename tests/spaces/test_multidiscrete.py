from copy import deepcopy

import numpy as np
import pytest

from gymnasium.spaces import Discrete, MultiDiscrete, flatten, unflatten
from gymnasium.utils.env_checker import data_equivalence


def test_multidiscrete_as_tuple():
    # 1D multi-discrete
    space = MultiDiscrete([3, 4, 5])

    assert space.shape == (3,)
    assert space[0] == Discrete(3)
    assert space[0:1] == MultiDiscrete([3])
    assert space[0:2] == MultiDiscrete([3, 4])
    assert space[:] == space and space[:] is not space

    # 2D multi-discrete
    space = MultiDiscrete([[3, 4, 5], [6, 7, 8]])

    assert space.shape == (2, 3)
    assert space[0, 1] == Discrete(4)
    assert space[0] == MultiDiscrete([3, 4, 5])
    assert space[0:1] == MultiDiscrete([[3, 4, 5]])
    assert space[0:2, :] == MultiDiscrete([[3, 4, 5], [6, 7, 8]])
    assert space[:, 0:1] == MultiDiscrete([[3], [6]])
    assert space[0:2, 0:2] == MultiDiscrete([[3, 4], [6, 7]])
    assert space[:] == space and space[:] is not space
    assert space[:, :] == space and space[:, :] is not space


def test_multidiscrete_start_as_tuple():
    # 1D multi-discrete
    space = MultiDiscrete([3, 4, 5], start=[10, 20, 30])

    assert space.shape == (3,)
    assert space[0] == Discrete(3, start=10)
    assert space[0:1] == MultiDiscrete([3], start=[10])
    assert space[0:2] == MultiDiscrete([3, 4], start=[10, 20])
    assert space[:] == space and space[:] is not space

    # 2D multi-discrete
    space = MultiDiscrete([[3, 4, 5], [6, 7, 8]], start=[[10, 20, 30], [40, 50, 60]])

    assert space.shape == (2, 3)
    assert space[0, 1] == Discrete(4, start=20)
    assert space[0] == MultiDiscrete([3, 4, 5], start=[10, 20, 30])
    assert space[0:1] == MultiDiscrete([[3, 4, 5]], start=[[10, 20, 30]])
    assert space[0:2, :] == MultiDiscrete(
        [[3, 4, 5], [6, 7, 8]], start=[[10, 20, 30], [40, 50, 60]]
    )
    assert space[:, 0:1] == MultiDiscrete([[3], [6]], start=[[10], [40]])
    assert space[0:2, 0:2] == MultiDiscrete(
        [[3, 4], [6, 7]], start=[[10, 20], [40, 50]]
    )
    assert space[:] == space and space[:] is not space
    assert space[:, :] == space and space[:, :] is not space


def test_multidiscrete_dtype_as_tuple():
    # 1D multi-discrete
    space = MultiDiscrete([3, 4, 5], dtype=np.int8)

    assert space.shape == (3,)
    assert space[0] == Discrete(3, dtype=np.int8)
    assert space[0:1] == MultiDiscrete([3], dtype=np.int8)
    assert space[0:2] == MultiDiscrete([3, 4], dtype=np.int8)
    assert space[:] == space and space[:] is not space

    # 2D multi-discrete
    space = MultiDiscrete([[3, 4, 5], [6, 7, 8]], dtype=np.uint32)

    assert space.shape == (2, 3)
    assert space[0, 1] == Discrete(4, dtype=np.uint32)
    assert space[0] == MultiDiscrete([3, 4, 5], dtype=np.uint32)
    assert space[0:1] == MultiDiscrete([[3, 4, 5]], dtype=np.uint32)
    assert space[0:2, :] == MultiDiscrete([[3, 4, 5], [6, 7, 8]], dtype=np.uint32)
    assert space[:, 0:1] == MultiDiscrete([[3], [6]], dtype=np.uint32)
    assert space[0:2, 0:2] == MultiDiscrete([[3, 4], [6, 7]], dtype=np.uint32)
    assert space[:] == space and space[:] is not space
    assert space[:, :] == space and space[:, :] is not space


def test_multidiscrete_subspace_reproducibility():
    # 1D multi-discrete
    space = MultiDiscrete([100, 200, 300])
    space.seed()

    assert data_equivalence(space[0].sample(), space[0].sample())
    assert data_equivalence(space[0:1].sample(), space[0:1].sample())
    assert data_equivalence(space[0:2].sample(), space[0:2].sample())
    assert data_equivalence(space[:].sample(), space[:].sample())
    assert data_equivalence(space[:].sample(), space.sample())

    # 2D multi-discrete
    space = MultiDiscrete([[300, 400, 500], [600, 700, 800]])
    space.seed()

    assert data_equivalence(space[0, 1].sample(), space[0, 1].sample())
    assert data_equivalence(space[0].sample(), space[0].sample())
    assert data_equivalence(space[0:1].sample(), space[0:1].sample())
    assert data_equivalence(space[0:2, :].sample(), space[0:2, :].sample())
    assert data_equivalence(space[:, 0:1].sample(), space[:, 0:1].sample())
    assert data_equivalence(space[0:2, 0:2].sample(), space[0:2, 0:2].sample())
    assert data_equivalence(space[:].sample(), space[:].sample())
    assert data_equivalence(space[:, :].sample(), space[:, :].sample())
    assert data_equivalence(space[:, :].sample(), space.sample())


def test_multidiscrete_start_subspace_reproducibility():
    # 1D multi-discrete
    space = MultiDiscrete([100, 200, 300], start=[-50, -100, -150])
    space.seed()

    assert data_equivalence(space[0].sample(), space[0].sample())
    assert data_equivalence(space[0:1].sample(), space[0:1].sample())
    assert data_equivalence(space[0:2].sample(), space[0:2].sample())
    assert data_equivalence(space[:].sample(), space[:].sample())
    assert data_equivalence(space[:].sample(), space.sample())

    # 2D multi-discrete
    space = MultiDiscrete(
        [[300, 400, 500], [600, 700, 800]],
        start=[[-150, -200, -250], [-300, -350, -400]],
    )
    space.seed()

    assert data_equivalence(space[0, 1].sample(), space[0, 1].sample())
    assert data_equivalence(space[0].sample(), space[0].sample())
    assert data_equivalence(space[0:1].sample(), space[0:1].sample())
    assert data_equivalence(space[0:2, :].sample(), space[0:2, :].sample())
    assert data_equivalence(space[:, 0:1].sample(), space[:, 0:1].sample())
    assert data_equivalence(space[0:2, 0:2].sample(), space[0:2, 0:2].sample())
    assert data_equivalence(space[:].sample(), space[:].sample())
    assert data_equivalence(space[:, :].sample(), space[:, :].sample())
    assert data_equivalence(space[:, :].sample(), space.sample())


def test_multidiscrete_length():
    space = MultiDiscrete(nvec=[3, 2, 4])
    assert len(space) == 3

    space = MultiDiscrete(nvec=[3, 2, 4], start=[10, 10, 10])
    assert len(space) == 3

    space = MultiDiscrete(nvec=[[2, 3], [3, 2]])
    with pytest.warns(
        UserWarning,
        match="Getting the length of a multi-dimensional MultiDiscrete space.",
    ):
        assert len(space) == 2

    space = MultiDiscrete(nvec=[[2, 3], [3, 2]], start=[[10, 20], [30, 40]])
    with pytest.warns(
        UserWarning,
        match="Getting the length of a multi-dimensional MultiDiscrete space.",
    ):
        assert len(space) == 2


def test_multidiscrete_integer_overflow():
    # Check if space can be flattened and unflattened without an integer overflow
    space = MultiDiscrete(nvec=[101, 101, 101, 101], dtype=np.int8)
    x = space.sample()
    y = flatten(space, x)
    z = unflatten(space, y)

    assert len(z) == 4
    assert np.array_equal(x, z)


def test_multidiscrete_start_contains():
    space = MultiDiscrete([3, 4, 5], start=[10, 20, 30])

    assert [10, 20, 30] in space
    assert [9, 20, 30] not in space

    assert [12, 23, 34] in space
    assert [13, 23, 34] not in space


def test_multidiscrete_equality():
    # Check if two spaces are equivalent.
    space_a = MultiDiscrete(nvec=[2, 3, 4], start=[0, 0, 1])

    space_b = MultiDiscrete(nvec=[2, 3, 4], start=[0, 0, 1])
    assert space_a == space_b

    space_b = MultiDiscrete(nvec=[2, 4, 3], start=[0, 0, 1])
    assert space_a != space_b

    space_b = MultiDiscrete(nvec=[2, 3, 4], start=[1, 0, 1])
    assert space_a != space_b

    space_b = MultiDiscrete(nvec=[2, 3, 4], start=[0, 1, 1])
    assert space_a != space_b

    space_b = MultiDiscrete(nvec=[2, 3, 4, 2], start=[1, 0, 0, 0])
    assert space_a != space_b


def test_space_legacy_pickling():
    """Test the legacy pickle of Discrete that is missing the `start` parameter."""
    # Test that start is corrected passed
    space = MultiDiscrete([1, 2, 3], start=[4, 5, 6])
    state = space.__dict__

    new_space = MultiDiscrete([1, 2, 3])
    new_space.__setstate__(state)
    assert new_space == space
    assert np.all(new_space.start == np.array([4, 5, 6]))

    legacy_space = MultiDiscrete([1, 2, 3])
    legacy_state = deepcopy(legacy_space.__dict__)
    del legacy_state["start"]

    new_legacy_space = MultiDiscrete([1, 2, 3])
    new_legacy_space.__setstate__(legacy_state)
    assert new_legacy_space == legacy_space
    assert np.all(new_legacy_space.start == np.array([0, 0, 0]))


def test_multidiscrete_sample_edge_cases():
    # Test edge case where one dimension has size 1
    space = MultiDiscrete([5, 1, 3])
    samples = [space.sample() for _ in range(1000)]
    samples = np.array(samples)

    # The second dimension should always be 0 (only one valid value)
    assert np.all(samples[:, 1] == 0)


def test_multidiscrete_sample():
    # Test sampling without a mask
    space = MultiDiscrete([5, 2, 3])
    samples = [space.sample() for _ in range(1000)]
    samples = np.array(samples)

    # Check that the samples fall within the bounds
    assert np.all(samples[:, 0] < 5)
    assert np.all(samples[:, 1] < 2)
    assert np.all(samples[:, 2] < 3)


def test_multidiscrete_sample_with_mask():
    # Test sampling with a mask
    space = MultiDiscrete([2, 3, 4])
    mask = (
        np.array([1, 0], dtype=np.int8),
        np.array([1, 1, 0], dtype=np.int8),
        np.array([1, 0, 1, 0], dtype=np.int8),
    )
    samples = [space.sample(mask=mask) for _ in range(1000)]
    assert all(sample in space for sample in samples)
    samples = np.array(samples)

    # Check that the samples respect the mask
    for i, dim in enumerate(space.nvec):
        for j in range(dim):
            if mask[i][j] == 0:
                assert np.all(samples[:, i] != j)


def test_multidiscrete_sample_probabilities():
    # Test sampling with probabilities
    space = MultiDiscrete([3, 3])
    probabilities = (
        np.array([0.1, 0.7, 0.2], dtype=np.float64),
        np.array([0.3, 0.3, 0.4], dtype=np.float64),
    )
    samples = [space.sample(probability=probabilities) for _ in range(10000)]
    assert all(sample in space for sample in samples)
    samples = np.array(samples)

    # Check empirical probabilities
    for i in range(2):
        counts = np.bincount(samples[:, i], minlength=3) / len(samples)
        np.testing.assert_allclose(counts, probabilities[i], atol=0.05)
