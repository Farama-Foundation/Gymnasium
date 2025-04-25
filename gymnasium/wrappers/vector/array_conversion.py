"""Vector wrapper for converting between Array API compatible frameworks."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from gymnasium.core import ActType, ObsType
from gymnasium.vector import VectorEnv, VectorWrapper
from gymnasium.vector.vector_env import ArrayType
from gymnasium.wrappers.array_conversion import Device, array_conversion


__all__ = ["ArrayConversion"]


class ArrayConversion(VectorWrapper):
    """Wraps a vector environment returning Array API compatible arrays so that it can be interacted with through a specific framework.

    Notes:
        A vectorized version of ``gymnasium.wrappers.ArrayConversion``

    Actions must be provided as Array API compatible arrays and observations, rewards, terminations and truncations will be returned in the desired framework.
    xp here is a module that is compatible with the Array API standard, e.g. ``numpy``, ``jax`` etc.

    Example:
        >>> import gymnasium as gym                                         # doctest: +SKIP
        >>> envs = gym.make_vec("JaxEnv-vx", 3)                             # doctest: +SKIP
        >>> envs = ArrayConversion(envs, xp=np)                                     # doctest: +SKIP
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
        super().__init__(env)
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
