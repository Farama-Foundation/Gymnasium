"""Helper functions and wrapper class for converting between PyTorch and NumPy."""
from __future__ import annotations

import functools
import numbers
from collections import abc
from typing import Any, Iterable, Mapping, SupportsFloat, Union

import numpy as np

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.error import DependencyNotInstalled


try:
    import torch

    Device = Union[str, torch.device]
except ImportError:
    raise DependencyNotInstalled(
        "Torch is not installed therefore cannot call `torch_to_numpy`, run `pip install torch`"
    )


__all__ = ["NumpyToTorchV0", "torch_to_numpy", "numpy_to_torch"]


@functools.singledispatch
def torch_to_numpy(value: Any) -> Any:
    """Converts a PyTorch Tensor into a NumPy Array."""
    raise Exception(
        f"No known conversion for Torch type ({type(value)}) to NumPy registered. Report as issue on github."
    )


@torch_to_numpy.register(numbers.Number)
@torch_to_numpy.register(torch.Tensor)
def _number_torch_to_numpy(value: numbers.Number | torch.Tensor) -> Any:
    """Convert a python number (int, float, complex) and torch.Tensor to a numpy array."""
    return np.array(value)


@torch_to_numpy.register(abc.Mapping)
def _mapping_torch_to_numpy(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Converts a mapping of PyTorch Tensors into a Dictionary of Jax Array."""
    return type(value)(**{k: torch_to_numpy(v) for k, v in value.items()})


@torch_to_numpy.register(abc.Iterable)
def _iterable_torch_to_numpy(value: Iterable[Any]) -> Iterable[Any]:
    """Converts an Iterable from PyTorch Tensors to an iterable of Jax Array."""
    return type(value)(torch_to_numpy(v) for v in value)


@functools.singledispatch
def numpy_to_torch(value: Any, device: Device | None = None) -> Any:
    """Converts a Jax Array into a PyTorch Tensor."""
    raise Exception(
        f"No known conversion for NumPy type ({type(value)}) to PyTorch registered. Report as issue on github."
    )


@numpy_to_torch.register(np.ndarray)
def _numpy_to_torch(value: np.ndarray, device: Device | None = None) -> torch.Tensor:
    """Converts a Jax Array into a PyTorch Tensor."""
    assert torch is not None
    tensor = torch.tensor(value)
    if device:
        return tensor.to(device=device)
    return tensor


@numpy_to_torch.register(abc.Mapping)
def _numpy_mapping_to_torch(
    value: Mapping[str, Any], device: Device | None = None
) -> Mapping[str, Any]:
    """Converts a mapping of Jax Array into a Dictionary of PyTorch Tensors."""
    return type(value)(**{k: numpy_to_torch(v, device) for k, v in value.items()})


@numpy_to_torch.register(abc.Iterable)
def _numpy_iterable_to_torch(
    value: Iterable[Any], device: Device | None = None
) -> Iterable[Any]:
    """Converts an Iterable from Jax Array to an iterable of PyTorch Tensors."""
    return type(value)(numpy_to_torch(v, device) for v in value)


class NumpyToTorchV0(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Wraps a numpy-based environment so that it can be interacted with through PyTorch Tensors.

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
        """Using a PyTorch based action that is converted to NumPy to be used by the environment.

        Args:
            action: A PyTorch-based action

        Returns:
            The PyTorch-based Tensor next observation, reward, termination, truncation, and extra info
        """
        jax_action = torch_to_numpy(action)
        obs, reward, terminated, truncated, info = self.env.step(jax_action)

        return (
            numpy_to_torch(obs, self.device),
            float(reward),
            bool(terminated),
            bool(truncated),
            numpy_to_torch(info, self.device),
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
            options = torch_to_numpy(options)

        return numpy_to_torch(self.env.reset(seed=seed, options=options), self.device)
