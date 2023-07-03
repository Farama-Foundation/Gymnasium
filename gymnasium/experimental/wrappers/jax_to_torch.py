# This wrapper will convert torch inputs for the actions and observations to Jax arrays
# for an underlying Jax environment then convert the return observations from Jax arrays
# back to torch tensors.
#
# Functionality for converting between torch and jax types originally copied from
# https://github.com/google/brax/blob/9d6b7ced2a13da0d074b5e9fbd3aad8311e26997/brax/io/torch.py
# Under the Apache 2.0 license. Copyright is held by the authors

"""Helper functions and wrapper class for converting between PyTorch and Jax."""
from __future__ import annotations

import functools
import numbers
from collections import abc
from typing import Any, Iterable, Mapping, SupportsFloat, Union

import gymnasium as gym
from gymnasium.core import RenderFrame, WrapperActType, WrapperObsType
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.wrappers.jax_to_numpy import jax_to_numpy


try:
    import jax
    import jax.numpy as jnp
    from jax import dlpack as jax_dlpack
except ImportError:
    raise DependencyNotInstalled(
        "Jax is not installed therefore cannot call `torch_to_jax`, run `pip install gymnasium[jax]`"
    )

try:
    import torch
    from torch.utils import dlpack as torch_dlpack

    Device = Union[str, torch.device]
except ImportError:
    raise DependencyNotInstalled(
        "Torch is not installed therefore cannot call `torch_to_jax`, run `pip install torch`"
    )


__all__ = ["JaxToTorchV0", "jax_to_torch", "torch_to_jax", "Device"]


@functools.singledispatch
def torch_to_jax(value: Any) -> Any:
    """Converts a PyTorch Tensor into a Jax Array."""
    raise Exception(
        f"No known conversion for Torch type ({type(value)}) to Jax registered. Report as issue on github."
    )


@torch_to_jax.register(numbers.Number)
def _number_torch_to_jax(value: numbers.Number) -> Any:
    """Convert a python number (int, float, complex) to a jax array."""
    return jnp.array(value)


@torch_to_jax.register(torch.Tensor)
def _tensor_torch_to_jax(value: torch.Tensor) -> jax.Array:
    """Converts a PyTorch Tensor into a Jax Array."""
    tensor = torch_dlpack.to_dlpack(value)  # pyright: ignore[reportPrivateImportUsage]
    tensor = jax_dlpack.from_dlpack(tensor)  # pyright: ignore[reportPrivateImportUsage]
    return tensor


@torch_to_jax.register(abc.Mapping)
def _mapping_torch_to_jax(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Converts a mapping of PyTorch Tensors into a Dictionary of Jax Array."""
    return type(value)(**{k: torch_to_jax(v) for k, v in value.items()})


@torch_to_jax.register(abc.Iterable)
def _iterable_torch_to_jax(value: Iterable[Any]) -> Iterable[Any]:
    """Converts an Iterable from PyTorch Tensors to an iterable of Jax Array."""
    return type(value)(torch_to_jax(v) for v in value)


@functools.singledispatch
def jax_to_torch(value: Any, device: Device | None = None) -> Any:
    """Converts a Jax Array into a PyTorch Tensor."""
    raise Exception(
        f"No known conversion for Jax type ({type(value)}) to PyTorch registered. Report as issue on github."
    )


@jax_to_torch.register(jax.Array)
def _devicearray_jax_to_torch(
    value: jax.Array, device: Device | None = None
) -> torch.Tensor:
    """Converts a Jax Array into a PyTorch Tensor."""
    assert jax_dlpack is not None and torch_dlpack is not None
    dlpack = jax_dlpack.to_dlpack(value)  # pyright: ignore[reportPrivateImportUsage]
    tensor = torch_dlpack.from_dlpack(dlpack)
    if device:
        return tensor.to(device=device)
    return tensor


@jax_to_torch.register(abc.Mapping)
def _jax_mapping_to_torch(
    value: Mapping[str, Any], device: Device | None = None
) -> Mapping[str, Any]:
    """Converts a mapping of Jax Array into a Dictionary of PyTorch Tensors."""
    return type(value)(**{k: jax_to_torch(v, device) for k, v in value.items()})


@jax_to_torch.register(abc.Iterable)
def _jax_iterable_to_torch(
    value: Iterable[Any], device: Device | None = None
) -> Iterable[Any]:
    """Converts an Iterable from Jax Array to an iterable of PyTorch Tensors."""
    return type(value)(jax_to_torch(v, device) for v in value)


class JaxToTorchV0(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Wraps a Jax-based environment so that it can be interacted with through PyTorch Tensors.

    Actions must be provided as PyTorch Tensors and observations will be returned as PyTorch Tensors.

    Note:
        For ``rendered`` this is returned as a NumPy array not a pytorch Tensor.
    """

    def __init__(self, env: gym.Env, device: Device | None = None):
        """Wrapper class to change inputs and outputs of environment to PyTorch tensors.

        Args:
            env: The Jax-based environment to wrap
            device: The device the torch Tensors should be moved to
        """
        gym.utils.RecordConstructorArgs.__init__(self, device=device)
        gym.Wrapper.__init__(self, env)

        self.device: Device | None = device

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict]:
        """Performs the given action within the environment.

        Args:
            action: The action to perform as a PyTorch Tensor

        Returns:
            The next observation, reward, termination, truncation, and extra info
        """
        jax_action = torch_to_jax(action)
        obs, reward, terminated, truncated, info = self.env.step(jax_action)

        return (
            jax_to_torch(obs, self.device),
            float(reward),
            bool(terminated),
            bool(truncated),
            jax_to_torch(info, self.device),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment returning PyTorch-based observation and info.

        Args:
            seed: The seed for resetting the environment
            options: The options for resetting the environment, these are converted to jax arrays.

        Returns:
            PyTorch-based observations and info
        """
        if options:
            options = torch_to_jax(options)

        return jax_to_torch(self.env.reset(seed=seed, options=options), self.device)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Returns the rendered frames as a NumPy array."""
        return jax_to_numpy(self.env.render())
