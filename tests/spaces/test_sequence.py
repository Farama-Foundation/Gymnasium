import re

import numpy as np
import pytest

import gymnasium as gym


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


@pytest.mark.parametrize("length", [3, np.int64(3), np.uint64(3)])
@pytest.mark.parametrize("sampling_argument", ["mask", "probability"])
@pytest.mark.parametrize("stack", [False, True])
def test_fixed_sample_length_scalar(length, sampling_argument, stack):
    """Fixed lengths preserve their scalar type and do not draw from the length RNG."""
    space = gym.spaces.Sequence(gym.spaces.Discrete(3), seed=0, stack=stack)
    rng_state = space.np_random.bit_generator.state

    assert space.generate_sample_length(length, sampling_argument) is length
    sample = space.sample(**{sampling_argument: (length, None)})
    assert len(sample) == length
    assert sample in space
    assert space.np_random.bit_generator.state == rng_state


@pytest.mark.parametrize("length", [True, np.bool_(True)])
@pytest.mark.parametrize("sampling_argument", ["mask", "probability"])
def test_boolean_sample_length_rejected(length, sampling_argument):
    """Boolean lengths remain invalid even though Python bool is an int subclass."""
    space = gym.spaces.Sequence(gym.spaces.Discrete(3), seed=0)
    with pytest.raises(TypeError, match="to be an integer or a np.ndarray"):
        space.sample(**{sampling_argument: (length, None)})
