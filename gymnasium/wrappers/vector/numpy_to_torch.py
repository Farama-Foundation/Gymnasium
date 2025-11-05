"""Wrapper for converting NumPy environments to PyTorch."""

from __future__ import annotations

import numpy as np
import torch

from gymnasium.vector import VectorEnv
from gymnasium.wrappers.numpy_to_torch import Device
from gymnasium.wrappers.vector.array_conversion import ArrayConversion


__all__ = ["NumpyToTorch"]


class NumpyToTorch(ArrayConversion):
    """Wraps a numpy-based environment so that it can be interacted with through PyTorch Tensors.

    Example:
        >>> import torch
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers.vector import NumpyToTorch
        >>> envs = gym.make_vec("CartPole-v1", 3)
        >>> envs = NumpyToTorch(envs)
        >>> obs, _ = envs.reset(seed=123)
        >>> type(obs)
        <class 'torch.Tensor'>
        >>> action = torch.tensor(envs.action_space.sample())
        >>> obs, reward, terminated, truncated, info = envs.step(action)
        >>> envs.close()
        >>> type(obs)
        <class 'torch.Tensor'>
        >>> type(reward)
        <class 'torch.Tensor'>
        >>> type(terminated)
        <class 'torch.Tensor'>
        >>> type(truncated)
        <class 'torch.Tensor'>
    """

    def __init__(self, env: VectorEnv, device: Device | None = None):
        """Wrapper class to change inputs and outputs of environment to PyTorch tensors.

        Args:
            env: The NumPy-based vector environment to wrap
            device: The device the torch Tensors should be moved to
        """
        super().__init__(env, env_xp=np, target_xp=torch, target_device=device)

        self.device: Device | None = device
