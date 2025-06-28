import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, Tuple
from gymnasium.utils.env_checker import data_equivalence


def test_sequence_inheritance():
    """The gymnasium Tuple space inherits from abc.Sequences, this test checks all functions work"""
    spaces = [Discrete(5), Discrete(10), Discrete(5)]
    tuple_space = Tuple(spaces)

    assert len(tuple_space) == len(spaces)
    # Test indexing
    for i in range(len(tuple_space)):
        assert tuple_space[i] == spaces[i]

    # Test iterable
    for space in tuple_space:
        assert space in spaces

    # Test count
    assert tuple_space.count(Discrete(5)) == 2
    assert tuple_space.count(Discrete(6)) == 0
    assert tuple_space.count(MultiBinary(2)) == 0

    # Test index
    assert tuple_space.index(Discrete(5)) == 0
    assert tuple_space.index(Discrete(5), 1) == 2

    # Test errors
    with pytest.raises(ValueError):
        tuple_space.index(Discrete(10), 0, 1)
    with pytest.raises(IndexError):
        assert tuple_space[4]


@pytest.mark.parametrize(
    "space, seed",
    [
        (Tuple([Discrete(5), Discrete(4)]), None),
        (Tuple([Discrete(5), Discrete(4)]), 123),
        (Tuple([Discrete(5), Discrete(4)]), (123, 456)),
        (
            Tuple(
                (Discrete(5), Tuple((Box(low=0.0, high=1.0, shape=(3,)), Discrete(2))))
            ),
            (123, (456, 789)),
        ),
        (
            Tuple(
                (
                    Discrete(3),
                    Dict(position=Box(low=0.0, high=1.0), velocity=Discrete(2)),
                )
            ),
            (123, {"position": 456, "velocity": 789}),
        ),
    ],
)
def test_seeds(space, seed):
    seeds1 = space.seed(seed)
    assert isinstance(seeds1, tuple)
    assert len(seeds1) == len(space)

    sample1 = space.sample()

    seeds2 = space.seed(seeds1)
    sample2 = space.sample()

    assert data_equivalence(seeds1, seeds2)
    assert data_equivalence(sample1, sample2)


@pytest.mark.parametrize(
    "space_fn",
    [
        lambda: Tuple(["abc"]),
        lambda: Tuple([gym.spaces.Box(0, 1), "abc"]),
        lambda: Tuple("abc"),
    ],
)
def test_bad_space_calls(space_fn):
    with pytest.raises(AssertionError):
        space_fn()


def test_contains_promotion():
    space = gym.spaces.Tuple((gym.spaces.Box(0, 1), gym.spaces.Box(-1, 0, (2,))))

    assert (
        np.array([0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    ) in space

    space = gym.spaces.Tuple((gym.spaces.Box(0, 1), gym.spaces.Box(-1, 0, (1,))))
    assert np.array([[0.0], [0.0]], dtype=np.float32) in space


def test_bad_seed():
    space = gym.spaces.Tuple((gym.spaces.Box(0, 1), gym.spaces.Box(0, 1)))
    with pytest.raises(
        TypeError,
        match="Expected seed type: list, tuple, int or None, actual type: <class 'float'>",
    ):
        space.seed(0.0)


def test_oneof_sample():
    """Tests the sample method with and without masks or probabilities."""
    space = gym.spaces.Tuple([Discrete(2), Box(-1, 1, shape=(2,))])

    # Unmasked sampling
    sample = space.sample()
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    assert space.spaces[0].contains(sample[0])
    assert space.spaces[1].contains(sample[1])

    # Masked sampling
    mask = (np.array([1, 0], dtype=np.int8), None)
    sample = space.sample(mask=mask)
    assert space.spaces[0].contains(sample[0])
    assert space.spaces[1].contains(sample[1])
    assert sample[0] == 0

    # Probability sampling
    probability = (np.array([0.8, 0.2], dtype=np.float64), None)
    samples_discrete = np.array(
        [space.sample(probability=probability)[0] for _ in range(1000)]
    )
    counts = np.bincount(samples_discrete, minlength=2) / len(samples_discrete)
    np.testing.assert_allclose(counts, probability[0], atol=0.05)


def test_invalid_sample_inputs():
    """Tests that invalid inputs to sample raise appropriate errors."""
    space = gym.spaces.Tuple([Discrete(2), Box(-1, 1, shape=(2,))])

    # Providing both mask and probability
    with pytest.raises(
        ValueError, match="Only one of `mask` or `probability` can be provided."
    ):
        space.sample(mask=(None, None), probability=(0.5, 0.5))

    # Invalid mask type
    with pytest.raises(
        AssertionError,
        match="Expected type of `mask` to be tuple, actual type: <class 'dict'>",
    ):
        space.sample(mask={"low": 0, "high": 1})

    # Invalid mask length
    with pytest.raises(
        AssertionError, match="Expected length of `mask` to be 2, actual length: 1"
    ):
        space.sample(mask=(None,))

    # Invalid probability length
    with pytest.raises(
        AssertionError,
        match="Expected length of `probability` to be 2, actual length: 1",
    ):
        space.sample(probability=(0.5,))

    # Invalid probability type
    with pytest.raises(
        AssertionError,
        match="Expected type of `probability` to be tuple, actual type: <class 'list'>",
    ):
        space.sample(probability=[0.5, 0.5])
