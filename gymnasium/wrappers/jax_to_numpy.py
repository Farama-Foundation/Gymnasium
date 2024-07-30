"""Helper functions and wrapper class for converting between numpy and Jax."""

from __future__ import annotations

import functools
import numbers
from collections import abc
from typing import Any, Iterable, Mapping, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame, WrapperActType, WrapperObsType
from gymnasium.error import DependencyNotInstalled


try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise DependencyNotInstalled(
        'Jax is not installed therefore cannot call `numpy_to_jax`, run `pip install "gymnasium[jax]"`'
    )

__all__ = ["JaxToNumpy", "jax_to_numpy", "numpy_to_jax"]


@functools.singledispatch
def numpy_to_jax(value: Any) -> Any:
    """Converts a value to a Jax Array."""
    raise Exception(
        f"No known conversion for Numpy type ({type(value)}) to Jax registered. Report as issue on github."
    )


@numpy_to_jax.register(numbers.Number)
def _number_to_jax(
    value: numbers.Number,
) -> jax.Array:
    """Converts a number (int, float, etc.) to a Jax Array."""
    assert jnp is not None
    return jnp.array(value)


@numpy_to_jax.register(np.ndarray)
def _numpy_array_to_jax(value: np.ndarray) -> jax.Array:
    """Converts a NumPy Array to a Jax Array with the same dtype (excluding float64 without being enabled)."""
    assert jnp is not None
    return jnp.array(value, dtype=value.dtype)


@numpy_to_jax.register(abc.Mapping)
def _mapping_numpy_to_jax(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Converts a dictionary of numpy arrays to a mapping of Jax Array."""
    return type(value)(**{k: numpy_to_jax(v) for k, v in value.items()})


@numpy_to_jax.register(abc.Iterable)
def _iterable_numpy_to_jax(
    value: Iterable[np.ndarray | Any],
) -> Iterable[jax.Array | Any]:
    """Converts an Iterable from Numpy Arrays to an iterable of Jax Array."""
    if hasattr(value, "_make"):
        # namedtuple - underline used to prevent potential name conflicts
        # noinspection PyProtectedMember
        return type(value)._make(numpy_to_jax(v) for v in value)
    else:
        return type(value)(numpy_to_jax(v) for v in value)


@functools.singledispatch
def jax_to_numpy(value: Any) -> Any:
    """Converts a value to a numpy array."""
    raise Exception(
        f"No known conversion for Jax type ({type(value)}) to NumPy registered. Report as issue on github."
    )


@jax_to_numpy.register(jax.Array)
def _devicearray_jax_to_numpy(value: jax.Array) -> np.ndarray:
    """Converts a Jax Array to a numpy array."""
    return np.array(value)


@jax_to_numpy.register(abc.Mapping)
def _mapping_jax_to_numpy(
    value: Mapping[str, jax.Array | Any]
) -> Mapping[str, np.ndarray | Any]:
    """Converts a dictionary of Jax Array to a mapping of numpy arrays."""
    return type(value)(**{k: jax_to_numpy(v) for k, v in value.items()})


@jax_to_numpy.register(abc.Iterable)
def _iterable_jax_to_numpy(
    value: Iterable[np.ndarray | Any],
) -> Iterable[jax.Array | Any]:
    """Converts an Iterable from Numpy arrays to an iterable of Jax Array."""
    if hasattr(value, "_make"):
        # namedtuple - underline used to prevent potential name conflicts
        # noinspection PyProtectedMember
        return type(value)._make(jax_to_numpy(v) for v in value)
    else:
        return type(value)(jax_to_numpy(v) for v in value)


class JaxToNumpy(
    gym.Wrapper[WrapperObsType, WrapperActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs,
):
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
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict]:
        """Transforms the action to a jax array .

        Args:
            action: the action to perform as a numpy array

        Returns:
            A tuple containing numpy versions of the next observation, reward, termination, truncation, and extra info.
        """
        jax_action = numpy_to_jax(action)
        obs, reward, terminated, truncated, info = self.env.step(jax_action)

        return (
            jax_to_numpy(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            jax_to_numpy(info),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
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

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Returns the rendered frames as a numpy array."""
        return jax_to_numpy(self.env.render())
