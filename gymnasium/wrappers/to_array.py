# This wrapper will convert torch inputs for the actions and observations to Jax arrays
# for an underlying Jax environment then convert the return observations from Jax arrays
# back to torch tensors.
#
# Functionality for converting between torch and jax types originally copied from
# https://github.com/google/brax/blob/9d6b7ced2a13da0d074b5e9fbd3aad8311e26997/brax/io/torch.py
# Under the Apache 2.0 license. Copyright is held by the authors

"""Helper functions and wrapper class for converting between arbitrary Array API compatible frameworks and a target framework."""

from __future__ import annotations

import functools
import numbers
from collections import abc
from typing import Any, Iterable, Mapping, SupportsFloat
from types import ModuleType
import gymnasium as gym
from gymnasium.core import RenderFrame, WrapperActType, WrapperObsType
from gymnasium.error import DependencyNotInstalled

try:
    from array_api_compat import numpy as numpy_namespace, is_array_api_obj

except ImportError:
    raise DependencyNotInstalled(
        'Array API packages are not installed therefore cannot call `to_array`, run `pip install "gymnasium[array-api]"`'
    )


__all__ = ["ToArray", "to_xp"]

# The NoneType is not defined in Python 3.9. Remove when the minimal version is bumped to >=3.10
_NoneType = type(None)
Array = Any  # TODO: Switch to ArrayAPI type once https://github.com/data-apis/array-api/pull/589 is merged
Device = Any  # TODO: Switch to ArrayAPI type if available


def module_namespace(module: ModuleType) -> ModuleType:
    """Determine the Array API compatible namespace of the given module."""
    if module.__name__ == "numpy":
        return numpy_namespace
    elif module.__name__ == "jax.numpy" or module.__name__ == "jax":
        import jax.numpy

        if hasattr(jax.numpy, "__array_api_version__"):
            jp = jax.numpy
        else:
            import jax.experimental.array_api as jp

        return jp
    else:
        raise ValueError(f"Unknown Array API framework: {module.__name__}")


@functools.singledispatch
def to_xp(value: Any, xp: ModuleType, device: Device | None = None) -> Any:
    """Converts a value into the specified xp module array type."""
    raise Exception(
        f"No known conversion for ({type(value)}) to xp module ({xp}) registered. Report as issue on github."
    )


@to_xp.register(numbers.Number)
def _number_to_xp(
    value: numbers.Number, xp: ModuleType, device: Device | None = None
) -> Array:
    """Convert a python number (int, float, complex) to an Array API framework array."""
    return xp.asarray(value, device=device)


@to_xp.register(abc.Mapping)
def _mapping_to_xp(
    value: Mapping[str, Any], xp: ModuleType, device: Device | None = None
) -> Mapping[str, Any]:
    """Converts a mapping of PyTorch Tensors into a Dictionary of Jax Array."""
    return type(value)(**{k: to_xp(v, xp, device) for k, v in value.items()})


@to_xp.register(abc.Iterable)
def _iterable_to_xp(
    value: Iterable[Any], xp: ModuleType, device: Device | None = None
) -> Iterable[Any]:
    """Converts an Iterable from PyTorch Tensors to an iterable of Jax Array."""
    if is_array_api_obj(value):
        # There is currently no type for ArrayAPI compatible objects, so they fall through to this
        # function registered for any Iterable. If they are arrays, we can convert them directly.
        return xp.asarray(value, device=device)
    if hasattr(value, "_make"):
        # namedtuple - underline used to prevent potential name conflicts
        # noinspection PyProtectedMember
        return type(value)._make(to_xp(v, xp, device) for v in value)
    else:
        return type(value)(to_xp(v, xp, device) for v in value)


@to_xp.register(_NoneType)
def _none_to_xp(value: None, xp: ModuleType, device: Device | None = None) -> None:
    """Passes through None values."""
    return value


class ToArray(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Wraps an Array API compatible environment so that it can be interacted with a specific Array API framework.

    Actions must be provided as Array API compatible arrays and observations will be returned as Arrays of the specified xp module.
    A vector version of the wrapper exists, :class:`gymnasium.wrappers.vector.ToArray`.

    Example:
        >>> import torch                                                # doctest: +SKIP
        >>> import gymnasium as gym                                     # doctest: +SKIP
        >>> env = gym.make("JaxEnv-vx")                                 # doctest: +SKIP
        >>> env = ToArray(env, xp=torch)                                # doctest: +SKIP
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
     * v1.0.0 - Initially added
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
        gym.utils.RecordConstructorArgs.__init__(
            self,
            env_device=env_device,
            target_device=target_device,
        )
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
        action = to_xp(action, xp=self._env_xp, device=self._env_device)
        obs, reward, terminated, truncated, info = self.env.step(action)

        return (
            to_xp(obs, xp=self._target_xp, device=self._target_device),
            float(reward),
            bool(terminated),
            bool(truncated),
            to_xp(info, xp=self._target_xp, device=self._target_device),
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
            options = to_xp(options, self._env_xp, self._env_device)

        return to_xp(
            self.env.reset(seed=seed, options=options),
            self._target_xp,
            self._target_device,
        )

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Returns the rendered frames as an xp Array."""
        return to_xp(self.env.render(), self._target_xp, self._target_device)
