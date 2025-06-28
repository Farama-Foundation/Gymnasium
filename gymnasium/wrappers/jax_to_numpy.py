"""Helper functions and wrapper class for converting between numpy and Jax."""

from __future__ import annotations

import functools

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled
from gymnasium.wrappers.array_conversion import (
    ArrayConversion,
    array_conversion,
    module_namespace,
)


try:
    import jax.numpy as jnp
except ImportError:
    raise DependencyNotInstalled(
        'Jax is not installed therefore cannot call `numpy_to_jax`, run `pip install "gymnasium[jax]"`'
    )

__all__ = ["JaxToNumpy", "jax_to_numpy", "numpy_to_jax"]


jax_to_numpy = functools.partial(array_conversion, xp=module_namespace(np))

numpy_to_jax = functools.partial(array_conversion, xp=module_namespace(jnp))


class JaxToNumpy(ArrayConversion):
    """Wraps a Jax-based environment such that it can be interacted with NumPy arrays.

    Actions must be provided as numpy arrays and observations will be returned as numpy arrays.
    A vector version of the wrapper exists, :class:`gymnasium.wrappers.vector.JaxToNumpy`.

    Notes:
        The Jax To Numpy and Numpy to Jax conversion does not guarantee a roundtrip (jax -> numpy -> jax) and vice versa.
        The reason for this is jax does not support non-array values, therefore numpy ``int_32(5) -> DeviceArray([5], dtype=jnp.int23)``

    Example:
        >>> import gymnasium as gym                                     # doctest: +SKIP
        >>> env = gym.make("JaxEnv-vx")                                 # doctest: +SKIP
        >>> env = JaxToNumpy(env)                                       # doctest: +SKIP
        >>> obs, _ = env.reset(seed=123)                                # doctest: +SKIP
        >>> type(obs)                                                   # doctest: +SKIP
        <class 'numpy.ndarray'>
        >>> action = env.action_space.sample()                          # doctest: +SKIP
        >>> obs, reward, terminated, truncated, info = env.step(action) # doctest: +SKIP
        >>> type(obs)                                                   # doctest: +SKIP
        <class 'numpy.ndarray'>
        >>> type(reward)                                                # doctest: +SKIP
        <class 'float'>
        >>> type(terminated)                                            # doctest: +SKIP
        <class 'bool'>
        >>> type(truncated)                                             # doctest: +SKIP
        <class 'bool'>

    Change logs:
     * v1.0.0 - Initially added
    """

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Wraps a jax environment such that the input and outputs are numpy arrays.

        Args:
            env: the jax environment to wrap
        """
        if jnp is None:
            raise DependencyNotInstalled(
                'Jax is not installed, run `pip install "gymnasium[jax]"`'
            )
        super().__init__(env=env, env_xp=jnp, target_xp=np)
