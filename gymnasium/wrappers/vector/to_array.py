"""Vector wrapper for converting between Array API compatible frameworks."""

from __future__ import annotations

from typing import Any
from types import ModuleType

from gymnasium.core import ActType, ObsType
from gymnasium.vector import VectorEnv, VectorWrapper
from gymnasium.vector.vector_env import ArrayType
from gymnasium.wrappers.to_array import to_xp, Device


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
        xp: ModuleType,
        env_device: Device | None = None,
        target_device: Device | None = None,
    ):
        """Wraps an environment such that the input and outputs are from the specified xp module.

        Args:
            env: The vector environment to wrap
            xp: An Array API compatible module, e.g. ``torch``, ``jax``, ``numpy``, ``cupy``, etc.
        """
        super().__init__(env, xp=xp, env_device=env_device, target_device=target_device)

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Transforms the action to the specified xp module array type.

        Args:
            actions: The action to perform

        Returns:
            A tuple containing xp versions of the next observation, reward, termination, truncation, and extra info.
        """
        actions = to_xp(actions, xp=self.xp, device=self.env_device)
        obs, reward, terminated, truncated, info = self.env.step(actions)

        return (
            to_xp(obs, xp=self.xp, device=self.target_device),
            to_xp(reward, xp=self.xp, device=self.target_device),
            to_xp(terminated, xp=self.xp, device=self.target_device),
            to_xp(truncated, xp=self.xp, device=self.target_device),
            to_xp(info, xp=self.xp, device=self.target_device),
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
            options = to_xp(options, xp=self.xp, device=self.env_device)

        return to_xp(
            self.env.reset(seed=seed, options=options),
            xp=self.xp,
            device=self.target_device,
        )
