import re
import warnings

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Box


@pytest.mark.parametrize(
    "box, expected_shape",
    [
        # Test with same 1-dim low and high shape
        (Box(low=np.zeros(2), high=np.ones(2)), (2,)),
        # Test with same multi-dim low and high shape
        (Box(low=np.zeros((2, 1)), high=np.ones((2, 1))), (2, 1)),
        # Test with scalar low high and different shape
        (Box(low=0, high=1, shape=(5, 2)), (5, 2)),
        (Box(low=0, high=1), (1,)),  # Test with int and int
        (Box(low=0.0, high=1.0), (1,)),  # Test with float and float
        (Box(low=np.zeros(1)[0], high=np.ones(1)[0]), (1,)),
        (Box(low=0.0, high=1), (1,)),  # Test with float and int
        (Box(low=0, high=np.int32(1)), (1,)),  # Test with python int and numpy int32
        (Box(low=0, high=np.ones(3)), (3,)),  # Test with array and scalar
        (Box(low=np.zeros(3), high=1.0), (3,)),  # Test with array and scalar
    ],
)
def test_shape_inference(box, expected_shape):
    """Test that the shape inference is as expected."""
    assert box.shape == expected_shape
    assert box.sample().shape == expected_shape


@pytest.mark.parametrize(
    "low, high, shape, message",
    [
        (
            0,
            1,
            (None,),
            "Expected all shape elements to be an integer, actual type: (<class 'NoneType'>,)",
        ),
        (
            0,
            1,
            (1, None),
            "Expected all shape elements to be an integer, actual type: (<class 'int'>, <class 'NoneType'>)",
        ),
        (
            0,
            1,
            (np.int64(1), None),
            "Expected all shape elements to be an integer, actual type: (<class 'numpy.int64'>, <class 'NoneType'>)",
        ),
        (
            np.zeros(3),
            np.ones(2),
            None,
            "high.shape doesn't match provided shape, high.shape: (2,), shape: (3,)",
        ),
        (
            np.zeros(2),
            np.ones(2),
            (3,),
            "low.shape doesn't match provided shape, low.shape: (2,), shape: (3,)",
        ),
    ],
)
def test_shape_errors(low, high, shape, message):
    """Test errors due to shape mismatch."""
    with pytest.raises(AssertionError, match=f"^{re.escape(message)}$"):
        Box(low=low, high=high, shape=shape)


@pytest.mark.parametrize(
    "dtype, error, message",
    [
        (
            None,
            AssertionError,
            "Box dtype must be explicitly provided, cannot be None.",
        ),
        (0, TypeError, "Cannot interpret '0' as a data type"),
        ("unknown", TypeError, "data type 'unknown' not understood"),
        (np.zeros(1), TypeError, "Cannot construct a dtype from an array"),
        # disabled datatypes
        (np.complex64, ValueError, "Invalid dtype (complex64) for Box"),
        (complex, ValueError, "Invalid dtype (complex128) for Box"),
        (object, ValueError, "Invalid dtype (object) for Box"),
        (str, ValueError, "Invalid dtype (<U0) for Box"),
    ],
)
def test_dtype_errors(dtype, error, message):
    """Test errors due to dtype mismatch either to being invalid or disallowed."""
    with pytest.raises(error, match=re.escape(message)):
        Box(low=0, high=1, dtype=dtype)


def test_bool_warning():
    """Test that using a bool dtype causes a warning."""
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Using `np.bool_` is not recommended with `Box`, `MultiBinary` is recommend for boolean samples."
        ),
    ):
        Box(low=False, high=True, dtype=np.bool_)

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Using `bool` is not recommended with `Box`, `MultiBinary` is recommend for boolean samples."
        ),
    ):
        Box(low=False, high=True, dtype=bool)


@pytest.mark.parametrize(
    "low, high, dtype",
    [
        # floats
        (0, 65501.0, np.float16),
        (-65499.0, 0, np.float16),
        # signed int
        (0, 32768, np.int16),
        (-32767, 0, np.int16),
        # unsigned int
        (-1, 100, np.uint8),
        (0, 300, np.uint8),
        (np.array([-1, 0]), np.array([0, 300]), np.uint8),
        (np.array([[-1], [0]]), np.array([[0], [300]]), np.uint8),
    ],
)
def test_out_of_bounds_error(low, high, dtype):
    with pytest.raises(
        ValueError, match="Box low or high value out of bounds of the dtype"
    ):
        Box(low=low, high=high, dtype=dtype)


@pytest.mark.parametrize(
    "low, high, dtype",
    [
        # Floats
        (np.nan, 0, np.float32),
        (0, np.nan, np.float32),
        ([0, np.nan], np.ones(2), np.float32),
        # Signed ints
        (np.nan, 0, np.int32),
        (0, np.nan, np.int32),
        ([0, np.nan], np.ones(2), np.int32),
        # Unsigned ints
        (-np.inf, 1, np.uint8),
        ([-np.inf, 0], 1, np.uint8),
        ([[-np.inf], [0]], 1, np.uint8),
        (0, np.inf, np.uint8),
        (0, [1, np.inf], np.uint8),
        (0, [[1], [np.inf]], np.uint8),
        (-np.inf, np.inf, np.uint8),
        ([0, -np.inf], [np.inf, 1], np.uint8),
    ],
)
def test_invalid_low_high(low, high, dtype):
    with pytest.raises(
        ValueError, match="Box low or high value are invalid for the dtype"
    ):
        Box(low=low, high=high, dtype=dtype)


@pytest.mark.parametrize(
    "low, high, dtype",
    [
        # floats
        (0, 1, np.float64),
        (0, 1, np.float32),
        (0, 1, np.float16),
        (np.zeros(2), np.ones(2), np.float32),
        (np.zeros(2), 1, np.float32),
        (-np.inf, 1, np.float32),
        (np.array([-np.inf, 0]), 1, np.float32),
        (0, np.inf, np.float32),
        (0, np.array([np.inf, 1]), np.float32),
        (-np.inf, np.inf, np.float32),
        (np.full((2,), -np.inf), np.full((2,), np.inf), np.float32),
        # signed ints
        (0, 1, np.int64),
        (0, 1, np.int32),
        (0, 1, np.int16),
        (0, 1, np.int8),
        (np.zeros(2), np.ones(2), np.int32),
        (np.zeros(2), 1, np.int32),
        (-np.inf, 1, np.int32),
        (np.array([-np.inf, 0]), 1, np.int32),
        (0, np.inf, np.int32),
        (0, np.array([np.inf, 1]), np.int32),
        # unsigned ints
        (0, 1, np.uint64),
        (0, 1, np.uint32),
        (0, 1, np.uint16),
        (0, 1, np.uint8),
    ],
)
def test_valid_low_high(low, high, dtype):
    with warnings.catch_warnings(record=True) as caught_warnings:
        space = Box(low=low, high=high, dtype=dtype)
        space.seed(0)

        sample = space.sample()
        assert space.contains(sample)

    assert len(caught_warnings) == 0, [x.message.args[0] for x in caught_warnings]


def test_contains_dtype():
    """Tests the Box contains function with different dtypes."""
    # Related Issues:
    # https://github.com/openai/gym/issues/2357
    # https://github.com/openai/gym/issues/2298

    space = Box(0, 1, (), dtype=np.float32)

    # casting will match the correct type
    assert np.array(0.5, dtype=np.float32) in space

    # float16 is in float32 space
    assert np.array(0.5, dtype=np.float16) in space

    # float64 is not in float32 space
    assert np.array(0.5, dtype=np.float64) not in space


@pytest.mark.parametrize(
    "lowhighshape",
    [
        dict(low=0, high=np.inf, shape=(2,)),
        dict(low=-np.inf, high=0, shape=(2,)),
        dict(low=-np.inf, high=np.inf, shape=(2,)),
        dict(low=0, high=np.inf, shape=(2, 3)),
        dict(low=-np.inf, high=0, shape=(2, 3)),
        dict(low=-np.inf, high=np.inf, shape=(2, 3)),
        dict(low=np.array([-np.inf, 0]), high=np.array([0.0, np.inf])),
    ],
)
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_infinite_space(lowhighshape, dtype):
    """
    To test spaces that are passed in have only 0 or infinite bounds because `space.high` and `space.low`
     are both modified within the init, we check for infinite when we know it's not 0
    """
    space = Box(**lowhighshape, dtype=dtype)

    assert np.all(space.low < space.high)

    # check that int bounds are bounded for everything but floats are unbounded for infinite
    assert space.is_bounded("above") is not np.any(space.high != 0)
    assert space.is_bounded("below") is not np.any(space.low != 0)
    assert space.is_bounded("both") is not (
        np.any(space.high != 0) | np.any(space.high != 0)
    )

    # check for dtype
    assert space.high.dtype == space.dtype
    assert space.low.dtype == space.dtype

    with pytest.raises(
        ValueError, match="manner is not in {'below', 'above', 'both'}, actual value:"
    ):
        space.is_bounded("test")

    # Check sample
    space.seed(0)
    sample = space.sample()

    # check if space contains sample
    assert sample in space

    # manually check that the sign of the sample is within the bounds
    assert np.all(np.sign(sample) <= np.sign(space.high))
    assert np.all(np.sign(space.low) <= np.sign(sample))


def test_legacy_state_pickling():
    legacy_state = {
        "dtype": np.dtype("float32"),
        "_shape": (5,),
        "low": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "high": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "bounded_below": np.array([True, True, True, True, True]),
        "bounded_above": np.array([True, True, True, True, True]),
        "_np_random": None,
    }

    b = Box(-1, 1, ())
    assert "low_repr" in b.__dict__ and "high_repr" in b.__dict__
    del b.__dict__["low_repr"]
    del b.__dict__["high_repr"]
    assert "low_repr" not in b.__dict__ and "high_repr" not in b.__dict__

    b.__setstate__(legacy_state)
    assert b.low_repr == "0.0"
    assert b.high_repr == "1.0"


def test_sample_mask():
    """Box cannot have a mask applied."""
    space = Box(0, 1)
    with pytest.raises(
        gym.error.Error,
        match=re.escape("Box.sample cannot be provided a mask, actual value: "),
    ):
        space.sample(mask=np.array([0, 1, 0], dtype=np.int8))
