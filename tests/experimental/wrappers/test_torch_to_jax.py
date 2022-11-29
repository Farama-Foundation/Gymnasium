import jax.numpy as jnp
import numpy as np
import pytest
import torch

from gymnasium.experimental.wrappers import JaxToTorchV0
from gymnasium.experimental.wrappers.torch_to_jax import jax_to_torch, torch_to_jax
from tests.testing_env import GenericTestEnv


def torch_data_equivalence(data_1, data_2) -> bool:
    if type(data_1) == type(data_2):
        if isinstance(data_1, dict):
            return data_1.keys() == data_2.keys() and all(
                torch_data_equivalence(data_1[k], data_2[k]) for k in data_1.keys()
            )
        elif isinstance(data_1, (tuple, list)):
            return len(data_1) == len(data_2) and all(
                torch_data_equivalence(o_1, o_2) for o_1, o_2 in zip(data_1, data_2)
            )
        elif isinstance(data_1, torch.Tensor):
            return data_1.shape == data_2.shape and np.allclose(
                data_1, data_2, atol=0.00001
            )
        else:
            return data_1 == data_2
    else:
        return False


@pytest.mark.parametrize(
    "value, expected_value",
    [
        (1.0, torch.tensor(1.0)),
        (2, torch.tensor(2)),
        ((3.0, 4), (torch.tensor(3.0), torch.tensor(4))),
        ([3.0, 4], [torch.tensor(3.0), torch.tensor(4)]),
        (
            {
                "a": 6.0,
                "b": 7,
            },
            {"a": torch.tensor(6.0), "b": torch.tensor(7)},
        ),
        (torch.tensor(1.0), torch.tensor(1.0)),
        (torch.tensor([1, 2]), torch.tensor([1, 2])),
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [2.0]])),
        (
            {"a": (1, torch.tensor(2.0), torch.tensor([3, 4])), "b": {"c": 5}},
            {
                "a": (torch.tensor(1), torch.tensor(2.0), torch.tensor([3, 4])),
                "b": {"c": torch.tensor(5)},
            },
        ),
    ],
)
def test_roundtripping(value, expected_value):
    """We test numpy -> jax -> numpy as this is direction in the NumpyToJax wrapper."""
    assert torch_data_equivalence(jax_to_torch(torch_to_jax(value)), expected_value)


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


def test_jax_to_torch():
    env = GenericTestEnv(reset_fn=jax_reset_func, step_fn=jax_step_func)

    # Check that the reset and step for jax environment are as expected
    obs, info = env.reset()
    assert isinstance(obs, jnp.DeviceArray)
    assert isinstance(info, dict) and isinstance(info["data"], jnp.DeviceArray)

    obs, reward, terminated, truncated, info = env.step(jnp.array([1, 2]))
    assert isinstance(obs, jnp.DeviceArray)
    assert isinstance(reward, jnp.DeviceArray)
    assert isinstance(terminated, jnp.DeviceArray) and isinstance(
        truncated, jnp.DeviceArray
    )
    assert isinstance(info, dict) and isinstance(info["data"], jnp.DeviceArray)

    # Check that the wrapped version is correct.
    wrapped_env = JaxToTorchV0(env)
    obs, info = wrapped_env.reset()
    assert isinstance(obs, torch.Tensor)
    assert isinstance(info, dict) and isinstance(info["data"], torch.Tensor)

    obs, reward, terminated, truncated, info = wrapped_env.step(torch.tensor([1, 2]))
    assert isinstance(obs, torch.Tensor)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict) and isinstance(info["data"], torch.Tensor)
