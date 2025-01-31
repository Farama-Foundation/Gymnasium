import numpy as np
import pytest

from gymnasium.spaces import Box, Discrete, MultiBinary, OneOf


def test_oneof_inheritance():
    """Tests that OneOf space properly inherits and implements required methods."""
    spaces = [Discrete(5), Box(-1, 1, shape=(3,)), MultiBinary(2)]
    oneof_space = OneOf(spaces)

    assert len(oneof_space) == len(spaces)
    # Test indexing
    for i in range(len(oneof_space)):
        assert oneof_space[i] == spaces[i]

    # Test iterable
    for space in oneof_space:
        assert space in spaces


@pytest.mark.parametrize(
    "spaces, seed",
    [
        ([Discrete(5), Box(-1, 1, shape=(3,))], None),
        ([Discrete(5), Box(-1, 1, shape=(3,))], 123),
        ([Discrete(5), Box(-1, 1, shape=(3,))], (123, 456, 789)),
    ],
)
def test_oneof_seeds(spaces, seed):
    oneof_space = OneOf(spaces)
    seeds = oneof_space.seed(seed)
    assert isinstance(seeds, tuple)
    assert len(seeds) == len(spaces) + 1


@pytest.mark.parametrize(
    "spaces_fn",
    [
        lambda: OneOf(["abc"]),
        lambda: OneOf([Box(0, 1), "abc"]),
        lambda: OneOf("abc"),
    ],
)
def test_bad_oneof_calls(spaces_fn):
    with pytest.raises(AssertionError):
        spaces_fn()


def test_oneof_contains():
    space = OneOf([Box(0, 1), Box(-1, 0, (2,))])

    assert (0, np.array([0.5], dtype=np.float32)) in space
    assert (1, np.array([-0.5, -0.5], dtype=np.float32)) in space

    assert (np.int64(0), np.array([0.5], dtype=np.float32)) in space

    assert (np.int32(0), np.array([0.5], dtype=np.float32)) not in space


def test_bad_oneof_seed():
    space = OneOf([Box(0, 1), Box(0, 1)])
    with pytest.raises(
        TypeError,
        match="Expected None, int, or tuple of ints, actual type: <class 'float'>",
    ):
        space.seed(0.0)


def test_oneof_sample():
    """Tests the sample method with and without masks or probabilities."""
    space = OneOf([Discrete(2), Box(-1, 1, shape=(2,))])

    # Unmasked sampling
    sample = space.sample()
    assert isinstance(sample, tuple)
    sample_idx, sample_value = sample
    assert sample_idx in [0, 1]
    assert sample_value in space.spaces[sample_idx]

    # Masked sampling
    mask = (np.array([1, 0], dtype=np.int8), None)
    sample_idx, sample_value = space.sample(mask=mask)
    assert sample_idx in [0, 1]
    while sample_idx != 0:
        sample_idx, sample_value = space.sample(mask=mask)
        if sample_idx == 0:
            assert sample_value == 0

    # Probability sampling
    probability = (np.array([0.8, 0.2], dtype=np.float64), None)
    sample_idx, sample_value = space.sample(probability=probability)
    assert sample_idx in [0, 1]


def test_invalid_sample_inputs():
    """Tests that invalid inputs to sample raise appropriate errors."""
    space = OneOf([Discrete(2), Box(-1, 1, shape=(2,))])

    # Providing both mask and probability
    with pytest.raises(
        ValueError, match="Only one of `mask` or `probability` can be provided."
    ):
        space.sample(mask=(None, None), probability=(0.5, 0.5))

    # Invalid mask type
    with pytest.raises(AssertionError, match="Expected type of `mask` is tuple"):
        space.sample(mask={"low": 0, "high": 1})

    # Invalid mask length
    with pytest.raises(AssertionError, match="Expected length of `mask` is 2"):
        space.sample(mask=(None,))

    # Invalid probability length
    with pytest.raises(AssertionError, match="Expected length of `probability` is 2"):
        space.sample(probability=(0.5,))

    # Invalid probability type
    with pytest.raises(AssertionError, match="Expected type of `probability` is tuple"):
        space.sample(probability=[0.5, 0.5])
