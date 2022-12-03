import jax.numpy as jnp
import numpy as np
import pytest

from gymnasium.experimental.wrappers import JaxToNumpyV0
from gymnasium.experimental.wrappers.numpy_to_jax import jax_to_numpy, numpy_to_jax
from gymnasium.utils.env_checker import data_equivalence
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize(
    "value, expected_value",
    [
        (1.0, np.array(1.0)),
        (2, np.array(2)),
        ((3.0, 4), (np.array(3.0), np.array(4))),
        ([3.0, 4], [np.array(3.0), np.array(4)]),
        (
            {
                "a": 6.0,
                "b": 7,
            },
            {"a": np.array(6.0), "b": np.array(7)},
        ),
        (np.array(1.0), np.array(1.0)),
        (np.array([1, 2]), np.array([1, 2])),
        (np.array([[1.0], [2.0]]), np.array([[1.0], [2.0]])),
        (
            {"a": (1, np.array(2.0), np.array([3, 4])), "b": {"c": 5}},
            {
                "a": (np.array(1), np.array(2.0), np.array([3, 4])),
                "b": {"c": np.array(5)},
            },
        ),
    ],
)
def test_roundtripping(value, expected_value):
    """We test numpy -> jax -> numpy as this is direction in the NumpyToJax wrapper."""
    assert data_equivalence(jax_to_numpy(numpy_to_jax(value)), expected_value)


def jax_reset_func(self, seed=None, options=None):
    return jnp.array([1.0, 2.0, 3.0]), {"data": jnp.array([1, 2, 3])}


def jax_step_func(self, action):
    assert isinstance(action, jnp.DeviceArray), type(action)
    return (
        jnp.array([1, 2, 3]),
        jnp.array(5.0),
        jnp.array(True),
        jnp.array(False),
        {"data": jnp.array([1.0, 2.0])},
    )


def test_jax_to_numpy():
    jax_env = GenericTestEnv(reset_fn=jax_reset_func, step_fn=jax_step_func)

    # Check that the reset and step for jax environment are as expected
    obs, info = jax_env.reset()
    assert isinstance(obs, jnp.DeviceArray)
    assert isinstance(info, dict) and isinstance(info["data"], jnp.DeviceArray)

    obs, reward, terminated, truncated, info = jax_env.step(jnp.array([1, 2]))
    assert isinstance(obs, jnp.DeviceArray)
    assert isinstance(reward, jnp.DeviceArray)
    assert isinstance(terminated, jnp.DeviceArray) and isinstance(
        truncated, jnp.DeviceArray
    )
    assert isinstance(info, dict) and isinstance(info["data"], jnp.DeviceArray)

    # Check that the wrapped version is correct.
    numpy_env = JaxToNumpyV0(jax_env)
    obs, info = numpy_env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict) and isinstance(info["data"], np.ndarray)

    obs, reward, terminated, truncated, info = numpy_env.step(np.array([1, 2]))
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict) and isinstance(info["data"], np.ndarray)
