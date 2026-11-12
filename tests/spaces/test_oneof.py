import json

import numpy as np
import pytest

from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    OneOf,
    Sequence,
    Text,
    Tuple,
)
from gymnasium.utils.env_checker import data_equivalence


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
    with pytest.raises(TypeError):
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


@pytest.mark.parametrize(
    "subspace",
    [
        Dict({"a": Box(0, 1, shape=(2,))}),
        Dict({"a": Box(0, 1, shape=(2,)), "b": Discrete(3)}),
        Tuple((Box(0, 1, shape=(2,)), Discrete(3))),
        Tuple((Discrete(3),)),
        Dict(
            a=Tuple((Box(0, 1, shape=(2,)), Dict(b=MultiBinary(2)))),
            c=Text(4),
        ),
        Sequence(Dict(a=Box(0, 1, shape=(2,))), stack=True),
        OneOf([Dict(a=Box(0, 1, shape=(2,))), Discrete(2)]),
    ],
)
def test_oneof_jsonable_composite_subspaces(subspace):
    """Tests that `to_jsonable` and `from_jsonable` round-trip for composite subspaces."""
    space = OneOf([subspace, Discrete(3)], seed=0)
    subspace.seed(1)
    samples = [
        (np.int64(0), subspace.sample()),
        (np.int64(1), np.int64(2)),
        (np.int64(0), subspace.sample()),
    ]

    jsonable = space.to_jsonable(samples)
    recovered = space.from_jsonable(json.loads(json.dumps(jsonable)))

    assert len(recovered) == len(samples)
    for sample, recovered_sample in zip(samples, recovered, strict=True):
        assert data_equivalence(sample, recovered_sample)
        assert recovered_sample in space


def test_oneof_jsonable_format():
    """Tests that the jsonable format stores one sample per element, keeping the format of previous versions."""
    space = OneOf([Box(0, 1, shape=(2,)), Dict(a=Discrete(3), b=Tuple((Discrete(2),)))])
    samples = [
        (np.int64(0), np.array([0.5, 0.25], dtype=np.float32)),
        (np.int64(1), {"a": np.int64(2), "b": (np.int64(1),)}),
    ]

    jsonable = space.to_jsonable(samples)
    assert jsonable == [[0, [0.5, 0.25]], [1, {"a": 2, "b": [1]}]]
    assert data_equivalence(space.from_jsonable(jsonable), samples)
