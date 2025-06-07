import re
import warnings

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Box


@pytest.mark.parametrize(
    "dtype, error, message",
    [
        (
            None,
            ValueError,
            "Box dtype must be explicitly provided, cannot be None.",
        ),
        (0, TypeError, "Cannot interpret '0' as a data type"),
        ("unknown", TypeError, "data type 'unknown' not understood"),
        (np.zeros(1), TypeError, "Cannot construct a dtype from an array"),
        # disabled datatypes
        (
            np.complex64,
            ValueError,
            "Invalid Box dtype (complex64), must be an integer, floating, or bool dtype",
        ),
        (
            complex,
            ValueError,
            "Invalid Box dtype (complex128), must be an integer, floating, or bool dtype",
        ),
        (
            object,
            ValueError,
            "Invalid Box dtype (object), must be an integer, floating, or bool dtype",
        ),
        (
            str,
            ValueError,
            "Invalid Box dtype (<U0), must be an integer, floating, or bool dtype",
        ),
    ],
)
def test_dtype_errors(dtype, error, message):
    """Test errors due to dtype mismatch either to being invalid or disallowed."""
    with pytest.raises(error, match=re.escape(message)):
        Box(low=0, high=1, dtype=dtype)


def _shape_inference_params():
    # Test with same 1-dim low and high shape
    yield Box(low=np.zeros(2), high=np.ones(2), dtype=np.float64), (2,)
    # Test with same multi-dim low and high shape
    yield Box(low=np.zeros((2, 1)), high=np.ones((2, 1)), dtype=np.float64), (2, 1)
    # Test with scalar low high and different shape
    yield Box(low=0, high=1, shape=(5, 2)), (5, 2)
    yield Box(low=0, high=1), (1,)  # Test with int and int
    yield Box(low=0.0, high=1.0), (1,)  # Test with float and float
    yield Box(low=np.zeros(1)[0], high=np.ones(1)[0]), (1,)
    yield Box(low=0.0, high=1), (1,)  # Test with float and int
    # Test with python int and numpy int32
    yield Box(low=0, high=np.int32(1)), (1,)
    # Test with array and scalar
    yield Box(low=0, high=np.ones(3), dtype=np.float64), (3,)
    yield Box(low=np.zeros(3), high=1.0, dtype=np.float64), (3,)


@pytest.mark.parametrize("box, expected_shape", _shape_inference_params())
def test_shape_inference(box, expected_shape):
    """Test that the shape inference is as expected."""
    assert box.shape == expected_shape
    assert box.sample().shape == expected_shape


@pytest.mark.parametrize(
    "low, high, shape, error_type, message",
    [
        (
            0,
            1,
            1,
            TypeError,
            "Expected Box shape to be an iterable, actual type=<class 'int'>",
        ),
        (
            0,
            1,
            (None,),
            TypeError,
            "Expected all Box shape elements to be integer, actual type=(<class 'NoneType'>,)",
        ),
        (
            0,
            1,
            (1, None),
            TypeError,
            "Expected all Box shape elements to be integer, actual type=(<class 'int'>, <class 'NoneType'>)",
        ),
        (
            0,
            1,
            (np.int64(1), None),
            TypeError,
            "Expected all Box shape elements to be integer, actual type=(<class 'numpy.int64'>, <class 'NoneType'>)",
        ),
        (
            np.zeros(3),
            np.ones(2),
            None,
            ValueError,
            "Box low.shape and high.shape don't match, low.shape=(3,), high.shape=(2,)",
        ),
        (
            np.zeros(2),
            np.ones(2),
            (3,),
            ValueError,
            "Box low.shape doesn't match provided shape, low.shape=(2,), shape=(3,)",
        ),
        (
            np.zeros(2),
            1,
            (3,),
            ValueError,
            "Box low.shape doesn't match provided shape, low.shape=(2,), shape=(3,)",
        ),
        (
            0,
            np.ones(2),
            (3,),
            ValueError,
            "Box high.shape doesn't match provided shape, high.shape=(2,), shape=(3,)",
        ),
    ],
)
def test_shape_errors(low, high, shape, error_type, message):
    """Test errors due to shape mismatch."""
    with pytest.raises(error_type, match=f"^{re.escape(message)}$"):
        Box(low=low, high=high, shape=shape, dtype=np.float64)


@pytest.mark.parametrize(
    "low, high, dtype",
    [
        # floats
        (0, 65505.0, np.float16),
        (-65505.0, 0, np.float16),
        # signed int
        (0, 32768, np.int16),
        (-32769, 0, np.int16),
        # unsigned int
        (-1, 100, np.uint8),
        (0, 300, np.uint8),
        # boolean
        (-1, 1, np.bool_),
        (0, 2, np.bool_),
        # array inputs
        (
            np.array([-1, 0]),
            np.array([0, 100]),
            np.uint8,
        ),
        (
            np.array([[-1], [0]]),
            np.array([[0], [100]]),
            np.uint8,
        ),
        (
            np.array([0, 0]),
            np.array([0, 300]),
            np.uint8,
        ),
        (
            np.array([[0], [0]]),
            np.array([[0], [300]]),
            np.uint8,
        ),
    ],
)
def test_out_of_bounds_error(low, high, dtype):
    with pytest.raises(
        ValueError, match=re.escape("is out of bounds of the dtype range,")
    ):
        Box(low=low, high=high, dtype=dtype)


@pytest.mark.parametrize(
    "low, high, dtype",
    [
        # Floats
        (np.nan, 0, np.float32),
        (0, np.nan, np.float32),
        (np.array([0, np.nan]), np.ones(2), np.float32),
        # Signed ints
        (np.nan, 0, np.int32),
        (0, np.nan, np.int32),
        (np.array([0, np.nan]), np.ones(2), np.int32),
        # Unsigned ints
        # (np.nan, 0, np.uint8),
        # (0, np.nan, np.uint8),
        # (np.array([0, np.nan]), np.ones(2), np.uint8),
        (-np.inf, 1, np.uint8),
        (np.array([-np.inf, 0]), 1, np.uint8),
        (0, np.inf, np.uint8),
        (0, np.array([1, np.inf]), np.uint8),
        # boolean
        (-np.inf, 1, np.bool_),
        (0, np.inf, np.bool_),
    ],
)
def test_invalid_low_high(low, high, dtype):
    if dtype == np.uint8 or dtype == np.bool_:
        with pytest.raises(
            ValueError, match=re.escape("Box unsigned int dtype don't support")
        ):
            Box(low=low, high=high, dtype=dtype)
    else:
        with pytest.raises(
            ValueError, match=re.escape("value can be equal to `np.nan`,")
        ):
            Box(low=low, high=high, dtype=dtype)


@pytest.mark.parametrize(
    "low, high, dtype",
    [
        # floats
        (0, 1, float),
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
        (0, 1, int),
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
        # boolean
        (0, 1, np.bool_),
    ],
)
def test_valid_low_high(low, high, dtype):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        space = Box(low=low, high=high, dtype=dtype)
        assert space.dtype == dtype
        assert space.low.dtype == dtype
        assert space.high.dtype == dtype

        space.seed(0)
        sample = space.sample()
        assert sample.dtype == dtype
        assert space.contains(sample)

    for warn in caught_warnings:
        if "precision lowered by casting to float32" not in warn.message.args[0]:
            raise Exception(warn)


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
    "low, high, shape",
    [
        (0, np.inf, (2,)),
        (-np.inf, 0, (2,)),
        (-np.inf, np.inf, (2,)),
        (0, np.inf, (2, 3)),
        (-np.inf, 0, (2, 3)),
        (-np.inf, np.inf, (2, 3)),
        (np.array([-np.inf, 0]), np.array([0.0, np.inf]), None),
    ],
)
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_infinite_space(low, high, shape, dtype):
    """
    To test spaces that are passed in have only 0 or infinite bounds because `space.high` and `space.low`
     are both modified within the init, we check for infinite when we know it's not 0
    """
    # Emits a warning for lowering the last example
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", UserWarning)
        space = Box(low=low, high=high, shape=shape, dtype=dtype)

    # Check if only the expected precision warning is emitted
    for warn in caught_warnings:
        if "precision lowered by casting to float32" not in warn.message.args[0]:
            raise Exception(warn)

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


def test_sample_probability_mask():
    """Box cannot have a probability mask applied."""
    space = Box(0, 1)
    with pytest.raises(
        gym.error.Error,
        match=re.escape(
            "Box.sample cannot be provided a probability mask, actual value: "
        ),
    ):
        space.sample(probability=np.array([0, 1, 0], dtype=np.float64))
