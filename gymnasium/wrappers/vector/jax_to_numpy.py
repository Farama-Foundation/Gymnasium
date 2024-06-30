"""Vector wrapper for converting between NumPy and Jax."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv, VectorWrapper
from gymnasium.vector.vector_env import ArrayType
from gymnasium.wrappers.jax_to_numpy import jax_to_numpy, numpy_to_jax


__all__ = ["JaxToNumpy"]


class JaxToNumpy(VectorWrapper):
    """Wraps a jax vector environment so that it can be interacted with through numpy arrays.

    Notes:
        A vectorized version of ``gymnasium.wrappers.JaxToNumpy``

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
        super().__init__(env)

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Transforms the action to a jax array .

        Args:
            actions: the action to perform as a numpy array

        Returns:
            A tuple containing numpy versions of the next observation, reward, termination, truncation, and extra info.
        """
        jax_actions = numpy_to_jax(actions)
        obs, reward, terminated, truncated, info = self.env.step(jax_actions)

        return (
            jax_to_numpy(obs),
            jax_to_numpy(reward),
            jax_to_numpy(terminated),
            jax_to_numpy(truncated),
            jax_to_numpy(info),
        )

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment returning numpy-based observation and info.

        Args:
            seed: The seed for resetting the environment
            options: The options for resetting the environment, these are converted to jax arrays.

        Returns:
            Numpy-based observations and info
        """
        if options:
            options = numpy_to_jax(options)

        return jax_to_numpy(self.env.reset(seed=seed, options=options))
