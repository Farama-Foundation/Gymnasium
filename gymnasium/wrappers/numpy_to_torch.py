"""Helper functions and wrapper class for converting between PyTorch and NumPy."""

from __future__ import annotations

import functools
from typing import Union

import numpy as np

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.wrappers.array_conversion import (
    ArrayConversion,
    array_conversion,
    module_namespace,
)


try:
    import torch

    Device = Union[str, torch.device]
except ImportError:
    raise DependencyNotInstalled(
        'Torch is not installed therefore cannot call `torch_to_numpy`, run `pip install "gymnasium[torch]"`'
    )


__all__ = ["NumpyToTorch", "torch_to_numpy", "numpy_to_torch", "Device"]


torch_to_numpy = functools.partial(array_conversion, xp=module_namespace(np))

numpy_to_torch = functools.partial(array_conversion, xp=module_namespace(torch))


class NumpyToTorch(ArrayConversion):
    """Wraps a NumPy-based environment such that it can be interacted with PyTorch Tensors.

    Actions must be provided as PyTorch Tensors and observations will be returned as PyTorch Tensors.
    A vector version of the wrapper exists, :class:`gymnasium.wrappers.vector.NumpyToTorch`.

    Note:
        For ``rendered`` this is returned as a NumPy array not a pytorch Tensor.

    Example:
        >>> import torch
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> env = NumpyToTorch(env)
        >>> obs, _ = env.reset(seed=123)
        >>> type(obs)
        <class 'torch.Tensor'>
        >>> action = torch.tensor(env.action_space.sample())
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> type(obs)
        <class 'torch.Tensor'>
        >>> type(reward)
        <class 'float'>
        >>> type(terminated)
        <class 'bool'>
        >>> type(truncated)
        <class 'bool'>

    Change logs:
     * v1.0.0 - Initially added
    """

    def __init__(self, env: gym.Env, device: Device | None = None):
        """Wrapper class to change inputs and outputs of environment to PyTorch tensors.

        Args:
            env: The NumPy-based environment to wrap
            device: The device the torch Tensors should be moved to
        """
        super().__init__(env=env, env_xp=np, target_xp=torch, target_device=device)

        self.device: Device | None = device
