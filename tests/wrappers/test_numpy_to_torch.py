"""Test suite for NumPyToTorch wrapper."""

import pickle
from typing import NamedTuple

import numpy as np
import pytest

import gymnasium


torch = pytest.importorskip("torch")


from gymnasium.utils.env_checker import data_equivalence  # noqa: E402
from gymnasium.wrappers.numpy_to_torch import (  # noqa: E402
    NumpyToTorch,
    numpy_to_torch,
    torch_to_numpy,
)
from tests.testing_env import GenericTestEnv  # noqa: E402


class ExampleNamedTuple(NamedTuple):
    a: np.ndarray
    b: np.ndarray


@pytest.mark.parametrize(
    "value, expected_value",
    [
        (1.0, np.array(1.0, dtype=np.float32)),
        (2, np.array(2, dtype=np.int64)),
        ((3.0, 4), (np.array(3.0, dtype=np.float32), np.array(4, dtype=np.int64))),
        ([3.0, 4], [np.array(3.0, dtype=np.float32), np.array(4, dtype=np.int64)]),
        (
            {
                "a": 6.0,
                "b": 7,
            },
            {"a": np.array(6.0, dtype=np.float32), "b": np.array(7, dtype=np.int64)},
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
                    np.array(1, dtype=np.int64),
                    np.array(2.0, dtype=np.float32),
                    np.array([3, 4], dtype=np.int32),
                ),
                "b": {"c": np.array(5, dtype=np.int64)},
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
    """We test numpy -> torch -> numpy as this is direction in the NumpyToTorch wrapper."""
    roundtripped_value = torch_to_numpy(numpy_to_torch(value))
    assert data_equivalence(roundtripped_value, expected_value)


def numpy_reset_func(self, seed=None, options=None):
    """A Numpy-based reset function."""
    return np.array([1.0, 2.0, 3.0]), {"data": np.array([1, 2, 3])}


def numpy_step_func(self, action):
    """A Numpy-based step function."""
    assert isinstance(action, np.ndarray), type(action)
    return (
        np.array([1, 2, 3]),
        5.0,
        True,
        False,
        {"data": np.array([1.0, 2.0])},
    )


def test_numpy_to_torch():
    """Tests the ``TorchToNumpy`` wrapper."""
    numpy_env = GenericTestEnv(reset_func=numpy_reset_func, step_func=numpy_step_func)
    obs, info = numpy_env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict) and isinstance(info["data"], np.ndarray)

    obs, reward, terminated, truncated, info = numpy_env.step(np.array([1, 2]))
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict) and isinstance(info["data"], np.ndarray)

    # Check that the wrapped version is correct.
    torch_env = NumpyToTorch(numpy_env)

    # Check that the reset and step for torch environment are as expected
    obs, info = torch_env.reset()
    assert isinstance(obs, torch.Tensor)
    assert isinstance(info, dict) and isinstance(info["data"], torch.Tensor)

    obs, reward, terminated, truncated, info = torch_env.step(torch.tensor([1, 2]))
    assert isinstance(obs, torch.Tensor)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict) and isinstance(info["data"], torch.Tensor)

    # Check that the wrapped environment can render. This implicitly returns None and requires a
    # None -> None conversion
    torch_env.render()

    # Test that the wrapped environment can be pickled
    env = gymnasium.make("CartPole-v1", disable_env_checker=True)
    wrapped_env = NumpyToTorch(env)
    pkl = pickle.dumps(wrapped_env)
    pickle.loads(pkl)
