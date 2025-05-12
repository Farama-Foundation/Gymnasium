"""Vector wrapper for converting between Array API compatible frameworks."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.vector import VectorEnv, VectorWrapper
from gymnasium.vector.vector_env import ArrayType
from gymnasium.wrappers.array_conversion import (
    Device,
    array_conversion,
    module_name_to_namespace,
)


__all__ = ["ArrayConversion"]


class ArrayConversion(VectorWrapper, gym.utils.RecordConstructorArgs):
    """Wraps a vector environment returning Array API compatible arrays so that it can be interacted with through a specific framework.

    Popular Array API frameworks include ``numpy``, ``torch``, ``jax.numpy``, ``cupy`` etc. With this wrapper, you can convert outputs from your environment to
    any of these frameworks. Conversely, actions are automatically mapped back to the environment framework, if possible without moving the
    data or device transfers.

    Notes:
        A vectorized version of :class:`gymnasium.wrappers.ArrayConversion`

    Example:
        >>> import gymnasium as gym                                         # doctest: +SKIP
        >>> envs = gym.make_vec("JaxEnv-vx", 3)                             # doctest: +SKIP
        >>> envs = ArrayConversion(envs, xp=np)                             # doctest: +SKIP
    """

    def __init__(
        self,
        env: VectorEnv,
        env_xp: ModuleType | str,
        target_xp: ModuleType | str,
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
        VectorWrapper.__init__(self, env)
        self._env_xp = env_xp
        self._target_xp = target_xp
        self._env_device = env_device
        self._target_device = target_device

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Transforms the action to the specified xp module array type.

        Args:
            actions: The action to perform

        Returns:
            A tuple containing xp versions of the next observation, reward, termination, truncation, and extra info.
        """
        actions = array_conversion(actions, xp=self._env_xp, device=self._env_device)
        obs, reward, terminated, truncated, info = self.env.step(actions)

        return (
            array_conversion(obs, xp=self._target_xp, device=self._target_device),
            array_conversion(reward, xp=self._target_xp, device=self._target_device),
            array_conversion(
                terminated, xp=self._target_xp, device=self._target_device
            ),
            array_conversion(truncated, xp=self._target_xp, device=self._target_device),
            array_conversion(info, xp=self._target_xp, device=self._target_device),
        )

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment returning xp-based observation and info.

        Args:
            seed: The seed for resetting the environment
            options: The options for resetting the environment, these are converted to xp arrays.

        Returns:
            xp-based observations and info
        """
        if options:
            options = array_conversion(
                options, xp=self._env_xp, device=self._env_device
            )

        return array_conversion(
            self.env.reset(seed=seed, options=options),
            xp=self._target_xp,
            device=self._target_device,
        )

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
