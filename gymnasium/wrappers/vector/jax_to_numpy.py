"""Vector wrapper for converting between NumPy and Jax."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.vector.array_conversion import ArrayConversion


__all__ = ["JaxToNumpy"]


class JaxToNumpy(ArrayConversion):
    """Wraps a jax vector environment so that it can be interacted with through numpy arrays.

    Notes:
        A vectorized version of :class:`gymnasium.wrappers.JaxToNumpy`

    Actions must be provided as numpy arrays and observations, rewards, terminations and truncations will be returned as numpy arrays.

    Example:
        >>> import gymnasium as gym                                         # doctest: +SKIP
        >>> envs = gym.make_vec("JaxEnv-vx", 3)                             # doctest: +SKIP
        >>> envs = JaxToNumpy(envs)                                         # doctest: +SKIP
    """

    def __init__(self, env: VectorEnv):
        """Wraps an environment such that the input and outputs are numpy arrays.

        Args:
            env: the vector jax environment to wrap
        """
        if jnp is None:
            raise DependencyNotInstalled(
                'Jax is not installed, run `pip install "gymnasium[jax]"`'
            )
        super().__init__(env, env_xp=jnp, target_xp=np)
