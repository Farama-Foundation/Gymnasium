"""Helper functions and wrapper class for converting between numpy and Jax."""
from __future__ import annotations

import functools
import numbers
from collections import abc
from typing import Any, Iterable, Mapping, SupportsFloat

import numpy as np

from gymnasium import Env, Wrapper
from gymnasium.core import RenderFrame, WrapperActType, WrapperObsType
from gymnasium.error import DependencyNotInstalled


try:
    import jax.numpy as jnp
except ImportError:
    # We handle the error internal to the relative functions
    jnp = None


@functools.singledispatch
def numpy_to_jax(value: Any) -> Any:
    """Converts a value to a Jax DeviceArray."""
    if jnp is None:
        raise DependencyNotInstalled(
            "Jax is not installed therefore cannot call `numpy_to_jax`, run `pip install gymnasium[jax]`"
        )
    else:
        raise Exception(
            f"No known conversion for Numpy type ({type(value)}) to Jax registered. Report as issue on github."
        )


if jnp is not None:

    @numpy_to_jax.register(numbers.Number)
    @numpy_to_jax.register(np.ndarray)
    def _number_ndarray_numpy_to_jax(
        value: np.ndarray | numbers.Number,
    ) -> jnp.DeviceArray:
        """Converts a numpy array or  number (int, float, etc.) to a Jax DeviceArray."""
        assert jnp is not None
        return jnp.array(value)

    @numpy_to_jax.register(abc.Mapping)
    def _mapping_numpy_to_jax(value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Converts a dictionary of numpy arrays to a mapping of Jax DeviceArrays."""
        return type(value)(**{k: numpy_to_jax(v) for k, v in value.items()})

    @numpy_to_jax.register(abc.Iterable)
    def _iterable_numpy_to_jax(
        value: Iterable[np.ndarray | Any],
    ) -> Iterable[jnp.DeviceArray | Any]:
        """Converts an Iterable from Numpy Arrays to an iterable of Jax DeviceArrays."""
        return type(value)(numpy_to_jax(v) for v in value)


@functools.singledispatch
def jax_to_numpy(value: Any) -> Any:
    """Converts a value to a numpy array."""
    if jnp is None:
        raise DependencyNotInstalled(
            "Jax is not installed therefore cannot call `jax_to_numpy`, run `pip install gymnasium[jax]`"
        )
    else:
        raise Exception(
            f"No known conversion for Jax type ({type(value)}) to NumPy registered. Report as issue on github."
        )


if jnp is not None:

    @jax_to_numpy.register(jnp.DeviceArray)
    def _devicearray_jax_to_numpy(value: jnp.DeviceArray) -> np.ndarray:
        """Converts a Jax DeviceArray to a numpy array."""
        return np.array(value)

    @jax_to_numpy.register(abc.Mapping)
    def _mapping_jax_to_numpy(
        value: Mapping[str, jnp.DeviceArray | Any]
    ) -> Mapping[str, np.ndarray | Any]:
        """Converts a dictionary of Jax DeviceArrays to a mapping of numpy arrays."""
        return type(value)(**{k: jax_to_numpy(v) for k, v in value.items()})

    @jax_to_numpy.register(abc.Iterable)
    def _iterable_jax_to_numpy(
        value: Iterable[np.ndarray | Any],
    ) -> Iterable[jnp.DeviceArray | Any]:
        """Converts an Iterable from Numpy arrays to an iterable of Jax DeviceArrays."""
        return type(value)(jax_to_numpy(v) for v in value)


class JaxToNumpyV0(Wrapper):
    """Wraps a jax environment so that it can be interacted with through numpy arrays.

    Actions must be provided as numpy arrays and observations will be returned as numpy arrays.

    Notes:
        The Jax To Numpy and Numpy to Jax conversion does not guarantee a roundtrip (jax -> numpy -> jax) and vice versa.
        The reason for this is jax does not support non-array values, therefore numpy ``int_32(5) -> DeviceArray([5], dtype=jnp.int23)``
    """

    def __init__(self, env: Env):
        """Wraps an environment such that the input and outputs are numpy arrays.

        Args:
            env: the environment to wrap
        """
        if jnp is None:
            raise DependencyNotInstalled(
                "jax is not installed, run `pip install gymnasium[jax]`"
            )
        super().__init__(env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict]:
        """Transforms the action to a jax array .

        Args:
            action: the action to perform as a numpy array

        Returns:
            A tuple containing the next observation, reward, termination, truncation, and extra info.
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
