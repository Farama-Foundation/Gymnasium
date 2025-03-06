"""Test suite for ToArray wrapper."""

from typing import NamedTuple
from itertools import product

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
array_api_compat = pytest.importorskip("array_api_compat")

from gymnasium.utils.env_checker import data_equivalence  # noqa: E402
from gymnasium.wrappers.to_array import ToArray, array_framework, to_xp  # noqa: E402
from tests.testing_env import GenericTestEnv  # noqa: E402

array_api_frameworks = (np, jax.numpy)
array_api_frameworks_combinations = [
    (s, t) for s, t in product(array_api_frameworks, repeat=2) if s != t
]


class ExampleNamedTuple(NamedTuple):
    a: jax.Array
    b: jax.Array


def value_parametrization():
    for xp, target_xp in array_api_frameworks_combinations:
        for value, expected_value in [
            (1.0, xp.asarray(1.0, dtype=xp.float32)),
            (2, xp.asarray(2, dtype=xp.int32)),
            (
                (3.0, 4),
                (xp.asarray(3.0, dtype=xp.float32), xp.asarray(4, dtype=xp.int32)),
            ),
            (
                [3.0, 4],
                [xp.asarray(3.0, dtype=xp.float32), xp.asarray(4, dtype=xp.int32)],
            ),
            (
                {
                    "a": 6.0,
                    "b": 7,
                },
                {
                    "a": xp.asarray(6.0, dtype=xp.float32),
                    "b": xp.asarray(7, dtype=xp.int32),
                },
            ),
            (xp.asarray(1.0, dtype=xp.float32), xp.asarray(1.0, dtype=xp.float32)),
            (xp.asarray(1.0, dtype=xp.uint8), xp.asarray(1.0, dtype=xp.uint8)),
            (xp.asarray([1, 2], dtype=xp.int32), xp.asarray([1, 2], dtype=xp.int32)),
            (
                xp.asarray([[1.0], [2.0]], dtype=xp.int32),
                xp.asarray([[1.0], [2.0]], dtype=xp.int32),
            ),
            (
                {
                    "a": (
                        1,
                        xp.asarray(2.0, dtype=xp.float32),
                        xp.asarray([3, 4], dtype=xp.int32),
                    ),
                    "b": {"c": 5},
                },
                {
                    "a": (
                        xp.asarray(1, dtype=xp.int32),
                        xp.asarray(2.0, dtype=xp.float32),
                        xp.asarray([3, 4], dtype=xp.int32),
                    ),
                    "b": {"c": xp.asarray(5, dtype=xp.int32)},
                },
            ),
            (
                ExampleNamedTuple(
                    a=xp.asarray([1, 2], dtype=xp.int32),
                    b=xp.asarray([1.0, 2.0], dtype=xp.float32),
                ),
                ExampleNamedTuple(
                    a=xp.asarray([1, 2], dtype=xp.int32),
                    b=xp.asarray([1.0, 2.0], dtype=xp.float32),
                ),
            ),
            (None, None),
        ]:
            yield (value, expected_value, xp, target_xp)


@pytest.mark.parametrize(
    "value,expected_value,source_xp,target_xp",
    value_parametrization(),
)
def test_roundtripping(value, expected_value, source_xp, target_xp):
    """We test numpy -> jax -> numpy as this is direction in the NumpyToJax wrapper.

    Warning: Jax doesn't support float64 out of the box, therefore, we only test float32 in this test.
    """
    source_xp = array_framework(source_xp)
    target_xp = array_framework(target_xp)
    roundtripped_value = to_xp(to_xp(value, xp=target_xp), xp=source_xp)
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
