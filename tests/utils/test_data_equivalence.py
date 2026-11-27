"""Tests for comparing arrays in environment observations and info."""

import numpy as np
import pytest

from gymnasium.spaces import Discrete
from gymnasium.utils.env_checker import check_env, data_equivalence
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize(
    "array",
    [
        np.array(["left", "right"]),
        np.array(["方向", "位置"]),
        np.array([b"left", b"right"]),
        np.array("label"),
        np.array([["a", "b"], ["c", "d"]]),
        np.array([], dtype="U4"),
        np.array([], dtype="S4"),
        np.array([("left", 1)], dtype=[("label", "U5"), ("value", "i4")]),
        np.array(["2026-01-01"], dtype="datetime64[D]"),
        np.array([1, 2], dtype="timedelta64[D]"),
    ],
)
def test_nonnumeric_array_equivalence(array, exact):
    """Non-numeric metadata uses equality rather than numeric tolerances."""
    assert data_equivalence(array, array.copy(), exact=exact)
    assert data_equivalence({"metadata": [array]}, {"metadata": [array.copy()]}, exact)
    assert not data_equivalence(array, np.expand_dims(array, 0), exact)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("dtype", ["U5", "S5"])
def test_string_array_differences(dtype, exact):
    """Different strings and dtypes remain unequal, even for empty arrays."""
    left = np.array(["left", "right"], dtype=dtype)
    right = np.array(["left", "wrong"], dtype=dtype)
    assert not data_equivalence(left, right, exact)
    assert not data_equivalence(left, left.astype(dtype[0] + "6"), exact)
    assert not data_equivalence(
        np.array([], dtype=dtype), np.array([], dtype="f8"), exact
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_numeric_array_tolerance(dtype):
    """Numeric arrays retain approximate and exact comparison modes."""
    left = np.array([1.0], dtype=dtype)
    close = np.array([1.0 + 1e-6], dtype=dtype)
    assert data_equivalence(left, close)
    assert not data_equivalence(left, close, exact=True)
    assert not data_equivalence(left, np.array([2.0], dtype=dtype))


@pytest.mark.parametrize("exact", [False, True])
def test_array_nan_and_object_semantics(exact):
    """NaNs stay unequal and object arrays still compare nested values recursively."""
    assert not data_equivalence(np.array([np.nan]), np.array([np.nan]), exact)
    assert not data_equivalence(
        np.array(["NaT"], dtype="datetime64[D]"),
        np.array(["NaT"], dtype="datetime64[D]"),
        exact,
    )
    left = np.empty(1, dtype=object)
    right = np.empty(1, dtype=object)
    left[0] = {"value": np.array([1.0])}
    right[0] = {"value": np.array([1.0 + 1e-6])}
    assert data_equivalence(left, right, exact) == (not exact)
    assert data_equivalence(np.array([True, False]), np.array([True, False]), exact)
    assert not data_equivalence(np.array([1, 2]), np.array([1, 3]), exact)


@pytest.mark.parametrize("dtype", ["U5", "S5"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_check_env_string_array_info(dtype, deterministic):
    """The full checker compares categorical info instead of raising a NumPy error."""
    calls = 0

    def step(self, action):
        nonlocal calls
        calls += 1
        label = "left" if deterministic or calls % 2 else "right"
        return (
            self.observation_space.sample(),
            0.0,
            False,
            False,
            {"label": np.array([label], dtype=dtype)},
        )

    env = GenericTestEnv(action_space=Discrete(2), step_func=step)
    if deterministic:
        check_env(env)
    else:
        with pytest.raises(
            AssertionError, match="Deterministic step info are not equivalent"
        ):
            check_env(env)
    env.close()
