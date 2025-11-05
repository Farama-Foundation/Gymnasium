# This wrapper will convert torch inputs for the actions and observations to Jax arrays
# for an underlying Jax environment then convert the return observations from Jax arrays
# back to torch tensors.
#
# Functionality for converting between torch and jax types originally copied from
# https://github.com/google/brax/blob/9d6b7ced2a13da0d074b5e9fbd3aad8311e26997/brax/io/torch.py
# Under the Apache 2.0 license. Copyright is held by the authors

"""Helper functions and wrapper class for converting between PyTorch and Jax."""

from __future__ import annotations

import functools
from typing import Union

import gymnasium as gym
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
        'Jax is not installed therefore cannot call `torch_to_jax`, run `pip install "gymnasium[jax]"`'
    )

try:
    import torch

    Device = Union[str, torch.device]
except ImportError:
    raise DependencyNotInstalled(
        'Torch is not installed therefore cannot call `torch_to_jax`, run `pip install "gymnasium[torch]"`'
    )


__all__ = ["JaxToTorch", "jax_to_torch", "torch_to_jax", "Device"]


torch_to_jax = functools.partial(array_conversion, xp=module_namespace(jnp))

jax_to_torch = functools.partial(array_conversion, xp=module_namespace(torch))


class JaxToTorch(ArrayConversion):
    """Wraps a Jax-based environment so that it can be interacted with PyTorch Tensors.

    Actions must be provided as PyTorch Tensors and observations will be returned as PyTorch Tensors.
    A vector version of the wrapper exists, :class:`gymnasium.wrappers.vector.JaxToTorch`.

    Note:
        For ``rendered`` this is returned as a NumPy array not a pytorch Tensor.

    Example:
        >>> import torch                                                # doctest: +SKIP
        >>> import gymnasium as gym                                     # doctest: +SKIP
        >>> env = gym.make("JaxEnv-vx")                                 # doctest: +SKIP
        >>> env = JaxtoTorch(env)                                       # doctest: +SKIP
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

    def __init__(self, env: gym.Env, device: Device | None = None):
        """Wrapper class to change inputs and outputs of environment to PyTorch tensors.

        Args:
            env: The Jax-based environment to wrap
            device: The device the torch Tensors should be moved to
        """
        super().__init__(env=env, env_xp=jnp, target_xp=torch, target_device=device)

        # TODO: Device was part of the public API, but should be removed in favor of _env_device and
        # _target_device.
        self.device: Device | None = device
