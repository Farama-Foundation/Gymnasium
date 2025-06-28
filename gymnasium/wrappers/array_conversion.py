# This wrapper will convert array inputs from an Array API compatible framework A for the actions
# to any other Array API compatible framework B for an underlying environment that is implemented
# in framework B, then convert the return observations from framework B back to framework A.
#
# More precisely, the wrapper will work for any two frameworks that can be made compatible with the
# `array-api-compat` package.
#
# See https://data-apis.org/array-api/latest/ for more information on the Array API standard, and
# https://data-apis.org/array-api-compat/ for more information on the Array API compatibility layer.
#
# General structure for converting between types originally copied from
# https://github.com/google/brax/blob/9d6b7ced2a13da0d074b5e9fbd3aad8311e26997/brax/io/torch.py
# Under the Apache 2.0 license. Copyright is held by the authors

"""Helper functions and wrapper class for converting between arbitrary Array API compatible frameworks and a target framework."""

from __future__ import annotations

import functools
import importlib
import numbers
from collections import abc
from collections.abc import Iterable, Mapping
from types import ModuleType, NoneType
from typing import Any, SupportsFloat

import numpy as np
from packaging.version import Version

import gymnasium as gym
from gymnasium.core import RenderFrame, WrapperActType, WrapperObsType
from gymnasium.error import DependencyNotInstalled


try:
    from array_api_compat import array_namespace, is_array_api_obj, to_device

except ImportError:
    raise DependencyNotInstalled(
        'Array API packages are not installed therefore cannot call `array_conversion`, run `pip install "gymnasium[array-api]"`'
    )


if Version(np.__version__) < Version("2.1.0"):
    raise DependencyNotInstalled("Array API functionality requires numpy >= 2.1.0")


__all__ = ["ArrayConversion", "array_conversion"]

Array = Any  # TODO: Switch to ArrayAPI type once https://github.com/data-apis/array-api/pull/589 is merged
Device = Any  # TODO: Switch to ArrayAPI type if available


def module_namespace(xp: ModuleType) -> ModuleType:
    """Determine the Array API compatible namespace of the given module.

    This function is closely linked to the `array_api_compat.array_namespace` function. It returns
    the compatible namespace for a module directly instead of from an array object of that module.

    See https://data-apis.org/array-api-compat/helper-functions.html#array_api_compat.array_namespace
    """
    try:
        return array_namespace(xp.empty(0))
    except AttributeError as e:
        raise ValueError(f"Module {xp} is not an Array API compatible module.") from e


def module_name_to_namespace(name: str) -> ModuleType:
    return module_namespace(importlib.import_module(name))


@functools.singledispatch
def array_conversion(value: Any, xp: ModuleType, device: Device | None = None) -> Any:
    """Convert a value into the specified xp module array type."""
    raise Exception(
        f"No known conversion for ({type(value)}) to xp module ({xp}) registered. Report as issue on github."
    )


@array_conversion.register(numbers.Number)
def _number_array_conversion(
    value: numbers.Number, xp: ModuleType, device: Device | None = None
) -> Array:
    """Convert a python number (int, float, complex) to an Array API framework array."""
    return xp.asarray(value, device=device)


@array_conversion.register(abc.Mapping)
def _mapping_array_conversion(
    value: Mapping[str, Any], xp: ModuleType, device: Device | None = None
) -> Mapping[str, Any]:
    """Convert a mapping of Arrays into a Dictionary of the specified xp module array type."""
    return type(value)(**{k: array_conversion(v, xp, device) for k, v in value.items()})


@array_conversion.register(abc.Iterable)
def _iterable_array_conversion(
    value: Iterable[Any], xp: ModuleType, device: Device | None = None
) -> Iterable[Any]:
    """Convert an Iterable from Arrays to an iterable of the specified xp module array type."""
    # There is currently no type for ArrayAPI compatible objects, so they fall through to this
    # function registered for any Iterable. If they are arrays, we can convert them directly.
    # We currently cannot pass the device to the from_dlpack function, since it is not supported
    # for some frameworks (see e.g. https://github.com/data-apis/array-api-compat/issues/204)
    if is_array_api_obj(value):
        return _array_api_array_conversion(value, xp, device)
    if hasattr(value, "_make"):
        # namedtuple - underline used to prevent potential name conflicts
        # noinspection PyProtectedMember
        return type(value)._make(array_conversion(v, xp, device) for v in value)
    return type(value)(array_conversion(v, xp, device) for v in value)


def _array_api_array_conversion(
    value: Array, xp: ModuleType, device: Device | None = None
) -> Array:
    """Convert an Array API compatible array to the specified xp module array type."""
    try:
        x = xp.from_dlpack(value)
        return to_device(x, device) if device is not None else x
    except (RuntimeError, BufferError):
        # If dlpack fails (e.g. because the array is read-only for frameworks that do not
        # support it), we create a copy of the array that we own and then convert it.
        # TODO: The correct treatment of read-only arrays is currently not fully clear in the
        # Array API. Once ongoing discussions are resolved, we should update this code to remove
        # any fallbacks.
        value_namespace = array_namespace(value)
        value_copy = value_namespace.asarray(value, copy=True)
        return xp.asarray(value_copy, device=device)


@array_conversion.register(NoneType)
def _none_array_conversion(
    value: None, xp: ModuleType, device: Device | None = None
) -> None:
    """Passes through None values."""
    return value


class ArrayConversion(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Wraps an Array API compatible environment so that it can be interacted with with another Array API framework.

    Popular Array API frameworks include ``numpy``, ``torch``, ``jax.numpy``, ``cupy`` etc. With this wrapper, you can convert outputs from your environment to
    any of these frameworks. Conversely, actions are automatically mapped back to the environment framework, if possible without moving the
    data or device transfers.

    A vector version of the wrapper exists, :class:`gymnasium.wrappers.vector.ArrayConversion`.

    Example:
        >>> import torch                                                # doctest: +SKIP
        >>> import jax.numpy as jnp                                     # doctest: +SKIP
        >>> import gymnasium as gym                                     # doctest: +SKIP
        >>> env = gym.make("JaxEnv-vx")                                 # doctest: +SKIP
        >>> env = ArrayConversion(env, env_xp=jnp, target_xp=torch)     # doctest: +SKIP
        >>> obs, _ = env.reset(seed=123)                                # doctest: +SKIP
        >>> type(obs)                                                   # doctest: +SKIP
        <class 'torch.Tensor'>
        >>> action = torch.tensor(env.action_space.sample())            # doctest: +SKIP
        >>> obs, reward, terminated, truncated, info = env.step(action) # doctest: +SKIP
        >>> type(obs)                                                   # doctest: +SKIP
        <class 'torch.Tensor'>
        >>> type(reward)                                                # doctest: +SKIP
        <class 'float'>
        >>> type(terminated)                                            # doctest: +SKIP
        <class 'bool'>
        >>> type(truncated)                                             # doctest: +SKIP
        <class 'bool'>

    Change logs:
     * v1.2.0 - Initially added
    """

    def __init__(
        self,
        env: gym.Env,
        env_xp: ModuleType,
        target_xp: ModuleType,
        env_device: Device | None = None,
        target_device: Device | None = None,
    ):
        """Wrapper class to change inputs and outputs of environment to any Array API framework.

        Args:
            env: The Array API compatible environment to wrap
            env_xp: The Array API framework the environment is on
            target_xp: The Array API framework to convert to
            env_device: The device the environment is on
            target_device: The device on which Arrays should be returned
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        self._env_xp = module_namespace(env_xp)
        self._target_xp = module_namespace(target_xp)
        self._env_device: Device | None = env_device
        self._target_device: Device | None = target_device

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict]:
        """Performs the given action within the environment.

        Args:
            action: The action to perform as any Array API compatible array

        Returns:
            The next observation, reward, termination, truncation, and extra info
        """
        action = array_conversion(action, xp=self._env_xp, device=self._env_device)
        obs, reward, terminated, truncated, info = self.env.step(action)

        return (
            array_conversion(obs, xp=self._target_xp, device=self._target_device),
            float(reward),
            bool(terminated),
            bool(truncated),
            array_conversion(info, xp=self._target_xp, device=self._target_device),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment returning observation and info as Array from any Array API compatible framework.

        Args:
            seed: The seed for resetting the environment
            options: The options for resetting the environment, these are converted to jax arrays.

        Returns:
            xp-based observations and info
        """
        if options:
            options = array_conversion(options, self._env_xp, self._env_device)

        return array_conversion(
            self.env.reset(seed=seed, options=options),
            self._target_xp,
            self._target_device,
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Returns the rendered frames as an xp Array."""
        return array_conversion(self.env.render(), self._target_xp, self._target_device)

    def __getstate__(self):
        """Returns the object pickle state with args and kwargs."""
        env_xp_name = self._env_xp.__name__.replace("array_api_compat.", "")
        target_xp_name = self._target_xp.__name__.replace("array_api_compat.", "")
        env_device = self._env_device
        target_device = self._target_device
        return {
            "env_xp_name": env_xp_name,
            "target_xp_name": target_xp_name,
            "env_device": env_device,
            "target_device": target_device,
            "env": self.env,
        }

    def __setstate__(self, d):
        """Sets the object pickle state using d."""
        self.env = d["env"]
        self._env_xp = module_name_to_namespace(d["env_xp_name"])
        self._target_xp = module_name_to_namespace(d["target_xp_name"])
        self._env_device = d["env_device"]
        self._target_device = d["target_device"]
