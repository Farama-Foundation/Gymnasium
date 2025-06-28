"""Test suite for TorchToJax wrapper."""

import pickle
from typing import NamedTuple

import pytest

import gymnasium


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")

from gymnasium.wrappers.jax_to_torch import (  # noqa: E402
    JaxToTorch,
    jax_to_torch,
    torch_to_jax,
)
from tests.testing_env import GenericTestEnv  # noqa: E402


def torch_data_equivalence(data_1, data_2) -> bool:
    """Return if two variables are equivalent that might contain ``torch.Tensor``."""
    if type(data_1) is type(data_2):
        if isinstance(data_1, dict):
            return data_1.keys() == data_2.keys() and all(
                torch_data_equivalence(data_1[k], data_2[k]) for k in data_1.keys()
            )
        elif isinstance(data_1, (tuple, list)):
            return len(data_1) == len(data_2) and all(
                torch_data_equivalence(o_1, o_2) for o_1, o_2 in zip(data_1, data_2)
            )
        elif isinstance(data_1, torch.Tensor):
            return data_1.shape == data_2.shape and torch.allclose(
                data_1, data_2, atol=0.00001
            )
        else:
            return data_1 == data_2
    else:
        return False


class ExampleNamedTuple(NamedTuple):
    a: torch.Tensor
    b: torch.Tensor


# Unless jax_enable_x64 is set at startup, jax will prevent us from creating double precision
# arrays. Therefore, all arrays are expected to be single precision after a roundtrip.
@pytest.mark.parametrize(
    "value, expected_value",
    [
        (1.0, torch.tensor(1.0)),
        (2, torch.tensor(2, dtype=torch.int32)),
        ((3.0, 4), (torch.tensor(3.0), torch.tensor(4, dtype=torch.int32))),
        ([3.0, 4], [torch.tensor(3.0), torch.tensor(4, dtype=torch.int32)]),
        (
            {
                "a": 6.0,
                "b": 7,
            },
            {"a": torch.tensor(6.0), "b": torch.tensor(7, dtype=torch.int32)},
        ),
        (torch.tensor(1.0), torch.tensor(1.0)),
        (torch.tensor(1), torch.tensor(1, dtype=torch.int32)),
        (torch.tensor([1, 2]), torch.tensor([1, 2], dtype=torch.int32)),
        (
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[1.0], [2.0]]),
        ),
        (
            {
                "a": (
                    1,
                    torch.tensor(2.0),
                    torch.tensor([3, 4]),
                ),
                "b": {"c": 5},
            },
            {
                "a": (
                    torch.tensor(1, dtype=torch.int32),
                    torch.tensor(2.0),
                    torch.tensor([3, 4], dtype=torch.int32),
                ),
                "b": {"c": torch.tensor(5, dtype=torch.int32)},
            },
        ),
        (
            ExampleNamedTuple(
                a=torch.tensor([1, 2]),
                b=torch.tensor([1.0, 2.0]),
            ),
            ExampleNamedTuple(
                a=torch.tensor([1, 2], dtype=torch.int32),
                b=torch.tensor([1.0, 2.0]),
            ),
        ),
        (None, None),
    ],
)
def test_roundtripping(value, expected_value):
    """We test numpy -> jax -> numpy as this is direction in the NumpyToJax wrapper."""
    roundtripped_value = jax_to_torch(torch_to_jax(value))
    assert torch_data_equivalence(roundtripped_value, expected_value)


def _jax_reset_func(self, seed=None, options=None):
    return jnp.array([1.0, 2.0, 3.0]), {"data": jnp.array([1, 2, 3])}


def _jax_step_func(self, action):
    assert isinstance(action, jax.Array), type(action)
    return (
        jnp.array([1, 2, 3]),
        jnp.array(5.0),
        jnp.array(True),
        jnp.array(False),
        {"data": jnp.array([1.0, 2.0])},
    )


def test_jax_to_torch_wrapper():
    """Tests the `JaxToTorchV0` wrapper."""
    env = GenericTestEnv(reset_func=_jax_reset_func, step_func=_jax_step_func)

    # Check that the reset and step for jax environment are as expected
    obs, info = env.reset()
    assert isinstance(obs, jax.Array)
    assert isinstance(info, dict) and isinstance(info["data"], jax.Array)

    obs, reward, terminated, truncated, info = env.step(jnp.array([1, 2]))
    assert isinstance(obs, jax.Array)
    assert isinstance(reward, jax.Array)
    assert isinstance(terminated, jax.Array) and isinstance(truncated, jax.Array)
    assert isinstance(info, dict) and isinstance(info["data"], jax.Array)

    # Check that the wrapped version is correct.
    wrapped_env = JaxToTorch(env)
    obs, info = wrapped_env.reset()
    assert isinstance(obs, torch.Tensor)
    assert isinstance(info, dict) and isinstance(info["data"], torch.Tensor)

    obs, reward, terminated, truncated, info = wrapped_env.step(torch.tensor([1, 2]))
    assert isinstance(obs, torch.Tensor)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict) and isinstance(info["data"], torch.Tensor)

    # Check that the wrapped environment can render. This implicitly returns None and requires  a
    # None -> None conversion
    wrapped_env.render()

    # Test that the wrapped environment can be pickled
    env = gymnasium.make("CartPole-v1", disable_env_checker=True)
    wrapped_env = JaxToTorch(env)
    pkl = pickle.dumps(wrapped_env)
    pickle.loads(pkl)
