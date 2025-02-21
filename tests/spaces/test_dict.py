import re
import warnings
from collections import OrderedDict

import numpy as np
import pytest

from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import data_equivalence


def test_dict_init():
    with pytest.raises(
        TypeError,
        match=r"^Unexpected Dict space input, expecting dict, OrderedDict or Sequence, actual type: ",
    ):
        Dict(Discrete(2))

    with pytest.raises(
        ValueError,
        match="Dict space keyword 'a' already exists in the spaces dictionary",
    ):
        Dict({"a": Discrete(3)}, a=Box(0, 1))

    with pytest.raises(
        AssertionError,
        match="Dict space element is not an instance of Space: key='b', space=Box",
    ):
        Dict(a=Discrete(2), b="Box")

    with warnings.catch_warnings(record=True) as caught_warnings:
        a = Dict({"a": Discrete(2), "b": Box(low=0.0, high=1.0)})
        b = Dict(OrderedDict(a=Discrete(2), b=Box(low=0.0, high=1.0)))
        c = Dict((("a", Discrete(2)), ("b", Box(low=0.0, high=1.0))))
        d = Dict(a=Discrete(2), b=Box(low=0.0, high=1.0))

        assert a == b == c == d
    assert len(caught_warnings) == 0

    # test sorting
    with warnings.catch_warnings(record=True) as caught_warnings:
        # Sorting is applied to the keys
        a = Dict({"b": Box(low=0.0, high=1.0), "a": Discrete(2)})
        assert a.keys() == {"a", "b"}

        # Sorting is not applied to the keys
        b = Dict(OrderedDict(b=Box(low=0.0, high=1.0), a=Discrete(2)))
        c = Dict((("b", Box(low=0.0, high=1.0)), ("a", Discrete(2))))
        d = Dict(b=Box(low=0.0, high=1.0), a=Discrete(2))
        assert b.keys() == c.keys() == d.keys() == {"b", "a"}
    assert len(caught_warnings) == 0

    # test sorting with different classes
    with warnings.catch_warnings(record=True) as caught_warnings:
        assert Dict({1: Discrete(2), "a": Discrete(3)}).keys() == {1, "a"}
    assert len(caught_warnings) == 0


DICT_SPACE = Dict(
    {
        "a": Box(low=0, high=1, shape=(3, 3)),
        "b": Dict(
            {
                "b_1": Box(low=-100, high=100, shape=(2,)),
                "b_2": Box(low=-1, high=1, shape=(2,)),
            }
        ),
        "c": Discrete(5),
    }
)


def test_dict_seeding():
    seeding_values = {
        "a": 0,
        "b": {
            "b_1": 1,
            "b_2": 2,
        },
        "c": 3,
    }
    seeded_values = DICT_SPACE.seed(seeding_values)
    assert data_equivalence(seeded_values, seeding_values)

    # "Unpack" the dict sub-spaces into individual spaces
    a = Box(low=0, high=1, shape=(3, 3), seed=0)
    b_1 = Box(low=-100, high=100, shape=(2,), seed=1)
    b_2 = Box(low=-1, high=1, shape=(2,), seed=2)
    c = Discrete(5, seed=3)

    for i in range(10):
        dict_sample = DICT_SPACE.sample()
        assert np.all(dict_sample["a"] == a.sample())
        assert np.all(dict_sample["b"]["b_1"] == b_1.sample())
        assert np.all(dict_sample["b"]["b_2"] == b_2.sample())
        assert dict_sample["c"] == c.sample()


def test_int_seeding():
    seeds = DICT_SPACE.seed(1)
    assert isinstance(seeds, dict)

    # rng, seeds = seeding.np_random(1)
    # subseeds = rng.choice(np.iinfo(int).max, size=3, replace=False)
    # b_rng, b_seeds = seeding.np_random(int(subseeds[1]))
    # b_subseeds = b_rng.choice(np.iinfo(int).max, size=2, replace=False)

    # "Unpack" the dict sub-spaces into individual spaces
    a = Box(low=0, high=1, shape=(3, 3), seed=seeds["a"])
    b_1 = Box(low=-100, high=100, shape=(2,), seed=seeds["b"]["b_1"])
    b_2 = Box(low=-1, high=1, shape=(2,), seed=seeds["b"]["b_2"])
    c = Discrete(5, seed=seeds["c"])

    for i in range(10):
        dict_sample = DICT_SPACE.sample()
        assert np.all(dict_sample["a"] == a.sample())
        assert np.all(dict_sample["b"]["b_1"] == b_1.sample())
        assert np.all(dict_sample["b"]["b_2"] == b_2.sample())
        assert dict_sample["c"] == c.sample()


def test_none_seeding():
    seeds = DICT_SPACE.seed(None)
    assert isinstance(seeds, dict)


def test_bad_seed():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Expected seed type: dict, int or None, actual type: <class 'str'>"
        ),
    ):
        DICT_SPACE.seed("a")


def test_mapping():
    """The Gym Dict space inherits from Mapping that allows it to appear like a standard python Dictionary."""
    assert len(DICT_SPACE) == 3

    a = DICT_SPACE["a"]
    b = Discrete(5)
    assert a != b
    DICT_SPACE["a"] = b
    assert DICT_SPACE["a"] == b

    with pytest.raises(
        AssertionError,
        match="Trying to set a to Dict space with value that is not a gymnasium space, actual type: <class 'int'>",
    ):
        DICT_SPACE["a"] = 5

    DICT_SPACE["a"] = a


def test_iterator():
    """Tests the Dict `__iter__` function correctly returns keys in the subspaces"""
    for key in DICT_SPACE:
        assert key in DICT_SPACE.spaces

    assert {key for key in DICT_SPACE} == DICT_SPACE.spaces.keys()


def test_keys_contains():
    """Test that `Dict.keys()` will correctly assess if the key is in the space."""
    space = Dict(a=Box(0, 1), b=Box(1, 2))

    for key in space.keys():
        assert key in space.keys()
    assert "a" in space.keys()

    assert "c" not in space.keys()


def test_sample_with_mask():
    """Test the sample method with valid masks."""
    space = Dict(
        {
            "a": Discrete(5),
            "b": Box(low=0, high=1, shape=(2,)),
        }
    )

    mask = {
        "a": np.array(
            [0, 1, 0, 0, 0], dtype=np.int8
        ),  # Only allow sampling the value 1
        "b": None,  # No mask for Box space
    }

    for _ in range(10):
        sample = space.sample(mask=mask)
        assert sample["a"] == 1  # Discrete space should only return 1
        assert space["b"].contains(sample["b"])


def test_sample_with_probability():
    """Test the sample method with valid probabilities."""
    space = Dict(
        {
            "a": Discrete(3),
            "b": Box(low=0, high=1, shape=(2,)),
        }
    )

    probability = {
        "a": np.array(
            [0.1, 0.7, 0.2], dtype=np.float64
        ),  # Sampling probabilities for Discrete space
        "b": None,  # No probability for Box space
    }

    samples = [space.sample(probability=probability)["a"] for _ in range(1000)]

    # Check that the sampling roughly follows the probability distribution
    counts = np.bincount(samples, minlength=3) / len(samples)
    np.testing.assert_almost_equal(counts, probability["a"], decimal=1)


def test_sample_with_invalid_mask():
    """Test the sample method with an invalid mask."""
    space = Dict(
        {
            "a": Discrete(5),
            "b": Box(low=0, high=1, shape=(2,)),
        }
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The expected shape of the sample mask is (5,), actual shape: (3,)"
        ),
    ):
        space.sample(
            mask={
                "a": np.array([1, 0, 0], dtype=np.int8),  # Length mismatch
                "b": None,
            }
        )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The expected dtype of the sample mask is np.int8, actual dtype: float32"
        ),
    ):
        space.sample(
            mask={
                "a": np.array([1, 0, 0, 1, 1], dtype=np.float32),  # dtype mismatch
                "b": None,
            }
        )


def test_sample_with_invalid_probability():
    """Test the sample method with an invalid probability."""
    space = Dict(
        {
            "a": Discrete(5),
            "b": Box(low=0, high=1, shape=(2,)),
        }
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The expected shape of the sample probability is (5,), actual shape: (2,)"
        ),
    ):
        space.sample(
            probability={
                "a": np.array([0.5, 0.5], dtype=np.float64),  # Length mismatch
                "b": None,
            }
        )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The expected dtype of the sample probability is np.float64, actual dtype: int8"
        ),
    ):
        space.sample(
            probability={
                "a": np.array([0.5, 0.5], dtype=np.int8),  # dtype mismatch
                "b": None,
            }
        )


def test_sample_with_mask_and_probability():
    """Ensure an error is raised when both mask and probability are provided."""
    space = Dict(
        {
            "a": Discrete(3),
            "b": Box(low=0, high=1, shape=(2,)),
        }
    )

    mask = {
        "a": np.array([1, 0, 1], dtype=np.int8),
        "b": None,
    }

    probability = {
        "a": np.array([0.5, 0.2, 0.3], dtype=np.float64),
        "b": None,
    }

    with pytest.raises(
        ValueError,
        match=re.escape("Only one of `mask` or `probability` can be provided"),
    ):
        space.sample(mask=mask, probability=probability)
