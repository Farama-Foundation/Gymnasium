"""Wrapper for converting NumPy environments to PyTorch."""
from __future__ import annotations

from typing import Any

from gymnasium.core import ActType, ObsType
from gymnasium.experimental.vector import VectorEnv, VectorWrapper
from gymnasium.experimental.vector.vector_env import ArrayType
from gymnasium.experimental.wrappers.jax_to_torch import Device
from gymnasium.experimental.wrappers.numpy_to_torch import (
    numpy_to_torch,
    torch_to_numpy,
)


__all__ = ["NumpyToTorchV0"]


class NumpyToTorchV0(VectorWrapper):
    """Wraps a numpy-based environment so that it can be interacted with through PyTorch Tensors."""

    def __init__(self, env: VectorEnv, device: Device | None = None):
        """Wrapper class to change inputs and outputs of environment to PyTorch tensors.

        Args:
            env: The Jax-based vector environment to wrap
            device: The device the torch Tensors should be moved to
        """
        super().__init__(env)

        self.device: Device | None = device

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Using a PyTorch based action that is converted to NumPy to be used by the environment.

        Args:
            action: A PyTorch-based action

        Returns:
            The PyTorch-based Tensor next observation, reward, termination, truncation, and extra info
        """
        jax_action = torch_to_numpy(actions)
        obs, reward, terminated, truncated, info = self.env.step(jax_action)

        return (
            numpy_to_torch(obs, self.device),
            numpy_to_torch(reward, self.device),
            numpy_to_torch(terminated, self.device),
            numpy_to_torch(truncated, self.device),
            numpy_to_torch(info, self.device),
        )

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
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
