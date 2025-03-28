"""Vector wrapper for converting between Array API compatible frameworks."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from gymnasium.core import ActType, ObsType
from gymnasium.vector import VectorEnv, VectorWrapper
from gymnasium.vector.vector_env import ArrayType
from gymnasium.wrappers.to_array import Device, to_xp


__all__ = ["ToArray"]


class ToArray(VectorWrapper):
    """Wraps a vector environment returning Array API compatible arrays so that it can be interacted with through a specific framework.

    Notes:
        A vectorized version of ``gymnasium.wrappers.ToArray``

    Actions must be provided as Array API compatible arrays and observations, rewards, terminations and truncations will be returned in the desired framework.
    xp here is a module that is compatible with the Array API standard, e.g. ``numpy``, ``jax`` etc.

    Example:
        >>> import gymnasium as gym                                         # doctest: +SKIP
        >>> envs = gym.make_vec("JaxEnv-vx", 3)                             # doctest: +SKIP
        >>> envs = ToArray(envs, xp=np)                                     # doctest: +SKIP
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
        actions = to_xp(actions, xp=self._env_xp, device=self._env_device)
        obs, reward, terminated, truncated, info = self.env.step(actions)

        return (
            to_xp(obs, xp=self._target_xp, device=self._target_device),
            to_xp(reward, xp=self._target_xp, device=self._target_device),
            to_xp(terminated, xp=self._target_xp, device=self._target_device),
            to_xp(truncated, xp=self._target_xp, device=self._target_device),
            to_xp(info, xp=self._target_xp, device=self._target_device),
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
            options = to_xp(options, xp=self._env_xp, device=self._env_device)

        return to_xp(
            self.env.reset(seed=seed, options=options),
            xp=self._target_xp,
            device=self._target_device,
        )
