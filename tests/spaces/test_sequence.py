import re

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces.utils import flatten, flatten_space, unflatten
from gymnasium.utils.env_checker import data_equivalence


def test_stacked_sequence():
    """Tests that a stacked sequence with a feature space of Box returns stacked values."""
    # Box
    space = gym.spaces.Sequence(gym.spaces.Box(0, 1, shape=(3,)), stack=True)
    sample = space.sample()
    # Check if the sample is in 2d format
    assert len(sample.shape) == 2

    # Discrete
    space = gym.spaces.Sequence(gym.spaces.Discrete(n=3), stack=True)
    sample = space.sample()
    # Check if the sample is a `np.ndarray` as supposed to a tuple
    assert type(sample) is np.ndarray


def test_sample():
    """Tests the sequence sampling works as expects and the errors are correctly raised."""
    space = gym.spaces.Sequence(gym.spaces.Box(0, 1))

    # Test integer mask length
    for length in range(4):
        sample = space.sample(mask=(length, None))
        assert sample in space
        assert len(sample) == length

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expects the length mask of `mask` to be greater than or equal to zero, actual value: -1"
        ),
    ):
        space.sample(mask=(-1, None))

    # Test np.array mask length
    sample = space.sample(mask=(np.array([5]), None))
    assert sample in space
    assert len(sample) == 5

    sample = space.sample(mask=(np.array([3, 4, 5]), None))
    assert sample in space
    assert len(sample) in [3, 4, 5]

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expects the shape of the length mask of `mask` to be 1-dimensional, actual shape: (2, 2)"
        ),
    ):
        space.sample(mask=(np.array([[2, 2], [2, 2]]), None))

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expects all values in the length_mask of `mask` to be greater than or equal to zero, actual values: [ 1  2 -1]"
        ),
    ):
        space.sample(mask=(np.array([1, 2, -1]), None))

    # Test with an invalid length
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Expects the type of length_mask of `mask` to be an integer or a np.ndarray, actual type: <class 'str'>"
        ),
    ):
        space.sample(mask=("abc", None))

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expects the shape of the length mask of `probability` to be 1-dimensional, actual shape: (2, 2)"
        ),
    ):
        space.sample(probability=(np.array([[2, 2], [2, 2]]), None))

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expects all values in the length_mask of `probability` to be greater than or equal to zero, actual values: [ 1  2 -1]"
        ),
    ):
        space.sample(probability=(np.array([1, 2, -1]), None))

    # Test with an invalid length
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Expects the type of length_mask of `probability` to be an integer or a np.ndarray, actual type: <class 'str'>"
        ),
    ):
        space.sample(probability=("abc", None))


def test_sample_with_mask():
    """Tests sampling with mask"""
    space = gym.spaces.Sequence(gym.spaces.Discrete(2))
    sample = space.sample(mask=(np.array([20]), np.array([0, 1], dtype=np.int8)))
    sample = np.array(sample)
    assert np.all(sample[:] == 1)
    assert np.all(value in space for value in sample)
    assert len(sample) == 20


def test_sample_with_probability():
    """Tests sampling with probability mask"""
    space = gym.spaces.Sequence(gym.spaces.Discrete(2))
    sample = space.sample(
        probability=(np.array([20]), np.array([0, 1], dtype=np.float64))
    )
    sample = np.array(sample)
    assert np.all(sample[:] == 1)
    assert np.all(value in space for value in sample)
    assert len(sample) == 20

    space = gym.spaces.Sequence(gym.spaces.Discrete(3))
    probability = (np.array([1000]), np.array([0, 0.2, 0.8], dtype=np.float64))
    sample = space.sample(probability=probability)
    sample = np.array(sample)
    assert np.all(np.isin(sample[:], [1, 2]))
    assert np.all(value in space for value in sample)
    counts = np.bincount(sample[:], minlength=3) / len(sample)
    np.testing.assert_allclose(counts, probability[1], atol=0.05)


EMPTY_STACKED_SEQUENCES = [
    pytest.param(
        gym.spaces.Box(-1, 1, shape=(), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0, 1), dtype=np.float32),
        id="scalar-box",
    ),
    pytest.param(
        gym.spaces.Box(-1, 1, shape=(2, 3), dtype=np.float32),
        np.empty((0, 2, 3), dtype=np.float32),
        np.empty((0, 6), dtype=np.float32),
        id="matrix-box",
    ),
    pytest.param(
        gym.spaces.Discrete(3, start=2, dtype=np.int16),
        np.empty((0,), dtype=np.int16),
        np.empty((0, 3), dtype=np.int16),
        id="discrete",
    ),
    pytest.param(
        gym.spaces.MultiDiscrete([[2, 3], [4, 2]], dtype=np.int32),
        np.empty((0, 2, 2), dtype=np.int32),
        np.empty((0, 11), dtype=np.int32),
        id="multidiscrete",
    ),
    pytest.param(
        gym.spaces.MultiBinary((2, 3)),
        np.empty((0, 2, 3), dtype=np.int8),
        np.empty((0, 6), dtype=np.int8),
        id="multibinary",
    ),
    pytest.param(
        gym.spaces.Dict(
            {
                "position": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
                "status": gym.spaces.Tuple(
                    (
                        gym.spaces.Discrete(3, dtype=np.int16),
                        gym.spaces.MultiBinary(2),
                    )
                ),
            }
        ),
        {
            "position": np.empty((0, 2), dtype=np.float32),
            "status": (
                np.empty((0,), dtype=np.int16),
                np.empty((0, 2), dtype=np.int8),
            ),
        },
        np.empty((0, 7), dtype=np.float32),
        id="nested-dict-tuple",
    ),
]


@pytest.mark.parametrize(
    "feature_space, expected, flat_expected", EMPTY_STACKED_SEQUENCES
)
@pytest.mark.parametrize("mask_type", ["mask", "probability"])
@pytest.mark.parametrize("length", [0, np.array([0])], ids=["integer", "array"])
def test_empty_stacked_sample(
    feature_space, expected, flat_expected, mask_type, length
):
    """Zero-length masks preserve the feature shape, dtype, and nested structure."""
    space = gym.spaces.Sequence(feature_space, stack=True)

    sample = space.sample(**{mask_type: (length, None)})

    assert sample in space
    assert data_equivalence(sample, expected, exact=True)


@pytest.mark.parametrize(
    "feature_space, expected, flat_expected", EMPTY_STACKED_SEQUENCES
)
@pytest.mark.parametrize(
    "transform", [flatten, unflatten], ids=["flatten", "unflatten"]
)
def test_empty_stacked_transform(feature_space, expected, flat_expected, transform):
    """Flatten and unflatten each accept valid empty samples without prior sampling."""
    space = gym.spaces.Sequence(feature_space, stack=True)
    flat_space = flatten_space(space)
    assert expected in space
    assert flat_expected in flat_space

    if transform is flatten:
        transformed = flatten(space, expected)
        assert transformed in flat_space
        assert data_equivalence(transformed, flat_expected, exact=True)
    else:
        transformed = unflatten(space, flat_expected)
        assert transformed in space
        assert data_equivalence(transformed, expected, exact=True)
