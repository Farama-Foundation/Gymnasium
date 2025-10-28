import re
from copy import deepcopy

import numpy as np
import pytest

from gymnasium.spaces import Discrete


def test_space_legacy_pickling():
    """Test the legacy pickle of Discrete that is missing the `start` parameter."""
    # Test that start is corrected passed
    space = Discrete(1, start=2)
    state = space.__dict__

    new_space = Discrete(1)
    new_space.__setstate__(state)
    assert new_space == space
    assert new_space.start == 2

    legacy_space = Discrete(1)
    legacy_state = deepcopy(legacy_space.__dict__)
    del legacy_state["start"]

    new_legacy_space = Discrete(2)
    new_legacy_space.__setstate__(legacy_state)
    assert new_legacy_space == legacy_space
    assert new_legacy_space.start == 0


def test_sample_mask():
    """Test that the mask parameter of the sample function works as expected."""
    space = Discrete(4, start=2)
    assert 2 <= space.sample() < 6
    assert space.sample(mask=np.array([0, 1, 0, 0], dtype=np.int8)) == 3
    assert space.sample(mask=np.array([0, 0, 0, 0], dtype=np.int8)) == 2
    assert space.sample(mask=np.array([0, 1, 0, 1], dtype=np.int8)) in [3, 5]


def test_probability_mask():
    """Test that the probability parameter of the sample function works as expected."""
    space = Discrete(4, start=2)
    assert space.sample(probability=np.array([0, 1, 0, 0], dtype=np.float64)) == 3
    assert space.sample(probability=np.array([0, 0.5, 0, 0.5], dtype=np.float64)) in [
        3,
        5,
    ]
    assert space.sample(
        probability=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ) in [
        2,
        3,
        4,
        5,
    ]


def test_sample_with_mask_and_probability():
    """Ensure an error is raised when both mask and probability are provided."""
    space = Discrete(4, start=2)

    with pytest.raises(
        ValueError,
        match=re.escape("Only one of `mask` or `probability` can be provided"),
    ):
        space.sample(
            mask=np.array([0, 1, 0, 0], dtype=np.int8),
            probability=np.array([0, 1, 0, 0], dtype=np.float64),
        )


def test_invalid_probability_mask_dtype():
    """Test that invalid probability mask dtype raises the correct exception."""
    space = Discrete(4, start=2)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The expected dtype of the sample probability is np.float64, actual dtype: int8"
        ),
    ):
        space.sample(probability=np.array([0, 1, 0, 0], dtype=np.int8))


def test_invalid_probability_mask_values():
    """Test that invalid probability mask values raises the correct exception."""
    space = Discrete(4, start=2)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "All values of the sample probability should be between 0 and 1, actual values: [-0.5  1.   0.5  0. ]"
        ),
    ):
        space.sample(probability=np.array([-0.5, 1, 0.5, 0], dtype=np.float64))

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The sum of the sample probability should be equal to 1, actual sum: 1.1"
        ),
    ):
        space.sample(probability=np.array([0.2, 0.3, 0.4, 0.2], dtype=np.float64))

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The sum of the sample probability should be equal to 1, actual sum: 0.0"
        ),
    ):
        space.sample(probability=np.array([0, 0, 0, 0], dtype=np.float64))


@pytest.mark.parametrize(
    "dtype, sample_dtype",
    [
        (int, np.int64),
        (np.int64, np.int64),
        (np.int32, np.int32),
        (np.uint8, np.uint8),
    ],
)
def test_dtype(dtype, sample_dtype):
    space = Discrete(n=5, dtype=dtype, start=2)

    sample = space.sample()
    sample_mask = space.sample(mask=np.array([0, 1, 0, 0, 0], dtype=np.int8))
    assert isinstance(sample, sample_dtype), type(sample)
    assert isinstance(sample_mask, sample_dtype), type(sample_mask)
    # test with probability
    sample_prob = space.sample(probability=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    assert isinstance(sample_prob, sample_dtype), type(sample_prob)


def test_equality():

    space = Discrete(n=3, dtype=np.int32)
    space_dif_dtype = Discrete(n=3)

    assert space != space_dif_dtype


@pytest.mark.parametrize(
    "element, dtype, expected_is_in",
    [
        (np.int32(0), np.int32, True),
        (np.int32(2), np.int32, False),
        (np.int64(-1), np.int32, False),
        (np.array(0, dtype=np.int8), np.int16, True),
        (np.array(0, dtype=np.int16), np.int8, False),
        (np.array(0), np.int32, False),
        (np.array(0), np.int64, True),
        (0, np.int16, True),
        (0, np.int32, True),
        (0, np.int64, True),
        (np.int32(0), np.int64, True),
        (np.int64(0), np.int16, False),
        (np.int32(0), np.int16, False),
        (np.int64(0), np.int8, False),
        (np.int8(0), np.int16, True),
    ],
)
def test_contains(element, dtype, expected_is_in):
    space = Discrete(3, start=-1, dtype=dtype)  # {-1, 0, 1}

    assert space.contains(element) == expected_is_in


@pytest.mark.parametrize(
    "dtype",
    [
        None,
        str,
        np.float32,
        np.complex64,
    ],
)
def test_dtype_error(dtype):
    with pytest.raises(TypeError, match="Invalid Discrete dtype"):
        Discrete(4, dtype=dtype)
