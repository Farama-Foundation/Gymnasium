"""Vector wrapper class for converting between PyTorch and Jax."""
from __future__ import annotations

from typing import Any

from gymnasium.core import ActType, ObsType
from gymnasium.experimental.vector import VectorEnv, VectorWrapper
from gymnasium.experimental.vector.vector_env import ArrayType
from gymnasium.experimental.wrappers.jax_to_torch import (
    Device,
    jax_to_torch,
    torch_to_jax,
)


__all__ = ["JaxToTorchV0"]


class JaxToTorchV0(VectorWrapper):
    """Wraps a Jax-based vector environment so that it can be interacted with through PyTorch Tensors.

    Actions must be provided as PyTorch Tensors and observations, rewards, terminations and truncations will be returned as PyTorch Tensors.
    """

    def __init__(self, env: VectorEnv, device: Device | None = None):
        """Vector wrapper to change inputs and outputs to PyTorch tensors.

        Args:
            env: The Jax-based vector environment to wrap
            device: The device the torch Tensors should be moved to
        """
        super().__init__(env)

        self.device: Device | None = device

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Performs the given action within the environment.

        Args:
            actions: The action to perform as a PyTorch Tensor

        Returns:
            Torch-based Tensors of the next observation, reward, termination, truncation, and extra info
        """
        jax_action = torch_to_jax(actions)
        obs, reward, terminated, truncated, info = self.env.step(jax_action)

        return (
            jax_to_torch(obs, self.device),
            jax_to_torch(reward, self.device),
            jax_to_torch(terminated, self.device),
            jax_to_torch(truncated, self.device),
            jax_to_torch(info, self.device),
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
            options = torch_to_jax(options)

        return jax_to_torch(self.env.reset(seed=seed, options=options), self.device)
