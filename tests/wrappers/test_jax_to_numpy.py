"""Test suite for JaxToNumpy wrapper."""

import pickle
from typing import NamedTuple

import numpy as np
import pytest

import gymnasium


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from gymnasium.utils.env_checker import data_equivalence  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import (  # noqa: E402
    JaxToNumpy,
    jax_to_numpy,
    numpy_to_jax,
)
from tests.testing_env import GenericTestEnv  # noqa: E402


class ExampleNamedTuple(NamedTuple):
    a: jax.Array
    b: jax.Array


@pytest.mark.parametrize(
    "value, expected_value",
    [
        (1.0, np.array(1.0, dtype=np.float32)),
        (2, np.array(2, dtype=np.int32)),
        ((3.0, 4), (np.array(3.0, dtype=np.float32), np.array(4, dtype=np.int32))),
        ([3.0, 4], [np.array(3.0, dtype=np.float32), np.array(4, dtype=np.int32)]),
        (
            {
                "a": 6.0,
                "b": 7,
            },
            {"a": np.array(6.0, dtype=np.float32), "b": np.array(7, dtype=np.int32)},
        ),
        (np.array(1.0, dtype=np.float32), np.array(1.0, dtype=np.float32)),
        (np.array(1.0, dtype=np.uint8), np.array(1.0, dtype=np.uint8)),
        (np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int32)),
        (
            np.array([[1.0], [2.0]], dtype=np.int32),
            np.array([[1.0], [2.0]], dtype=np.int32),
        ),
        (
            {
                "a": (
                    1,
                    np.array(2.0, dtype=np.float32),
                    np.array([3, 4], dtype=np.int32),
                ),
                "b": {"c": 5},
            },
            {
                "a": (
                    np.array(1, dtype=np.int32),
                    np.array(2.0, dtype=np.float32),
                    np.array([3, 4], dtype=np.int32),
                ),
                "b": {"c": np.array(5, dtype=np.int32)},
            },
        ),
        (
            ExampleNamedTuple(
                a=np.array([1, 2], dtype=np.int32),
                b=np.array([1.0, 2.0], dtype=np.float32),
            ),
            ExampleNamedTuple(
                a=np.array([1, 2], dtype=np.int32),
                b=np.array([1.0, 2.0], dtype=np.float32),
            ),
        ),
        (None, None),
    ],
)
def test_roundtripping(value, expected_value):
    """We test numpy -> jax -> numpy as this is direction in the NumpyToJax wrapper.

    Warning: Jax doesn't support float64 out of the box, therefore, we only test float32 in this test.
    """
    roundtripped_value = jax_to_numpy(numpy_to_jax(value))
    assert data_equivalence(roundtripped_value, expected_value)


def jax_reset_func(self, seed=None, options=None):
    """A jax-based reset function."""
    return jnp.array([1.0, 2.0, 3.0]), {"data": jnp.array([1, 2, 3])}


def jax_step_func(self, action):
    """A jax-based step function."""
    assert isinstance(action, jax.Array), type(action)
    return (
        jnp.array([1, 2, 3]),
        jnp.array(5.0),
        jnp.array(True),
        jnp.array(False),
        {"data": jnp.array([1.0, 2.0])},
    )


def test_jax_to_numpy_wrapper():
    """Tests the ``JaxToNumpyV0`` wrapper."""
    jax_env = GenericTestEnv(reset_func=jax_reset_func, step_func=jax_step_func)

    # Check that the reset and step for jax environment are as expected
    obs, info = jax_env.reset()
    assert isinstance(obs, jax.Array)
    assert isinstance(info, dict) and isinstance(info["data"], jax.Array)

    obs, reward, terminated, truncated, info = jax_env.step(jnp.array([1, 2]))
    assert isinstance(obs, jax.Array)
    assert isinstance(reward, jax.Array)
    assert isinstance(terminated, jax.Array) and isinstance(truncated, jax.Array)
    assert isinstance(info, dict) and isinstance(info["data"], jax.Array)

    # Check that the wrapped version is correct.
    numpy_env = JaxToNumpy(jax_env)
    obs, info = numpy_env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict) and isinstance(info["data"], np.ndarray)

    obs, reward, terminated, truncated, info = numpy_env.step(
        np.array([1, 2], dtype=np.int32)
    )
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict) and isinstance(info["data"], np.ndarray)

    # Check that the wrapped environment can render. This implicitly returns None and requires  a
    # None -> None conversion
    numpy_env.render()

    # Test that the wrapped environment can be pickled
    env = gymnasium.make("CartPole-v1", disable_env_checker=True)
    wrapped_env = JaxToNumpy(env)
    pkl = pickle.dumps(wrapped_env)
    pickle.loads(pkl)
