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
