"""Vector wrapper class for converting between PyTorch and Jax."""

from __future__ import annotations

import jax.numpy as jnp
import torch

from gymnasium.vector import VectorEnv
from gymnasium.wrappers.jax_to_torch import Device
from gymnasium.wrappers.vector.array_conversion import ArrayConversion


__all__ = ["JaxToTorch"]


class JaxToTorch(ArrayConversion):
    """Wraps a Jax-based vector environment so that it can be interacted with through PyTorch Tensors.

    Actions must be provided as PyTorch Tensors and observations, rewards, terminations and truncations will be returned as PyTorch Tensors.

    Example:
        >>> import gymnasium as gym                                         # doctest: +SKIP
        >>> envs = gym.make_vec("JaxEnv-vx", 3)                             # doctest: +SKIP
        >>> envs = JaxToTorch(envs)                                         # doctest: +SKIP
    """

    def __init__(self, env: VectorEnv, device: Device | None = None):
        """Vector wrapper to change inputs and outputs to PyTorch tensors.

        Args:
            env: The Jax-based vector environment to wrap
            device: The device the torch Tensors should be moved to
        """
        super().__init__(env, env_xp=jnp, target_xp=torch, target_device=device)

        self.device: Device | None = device
