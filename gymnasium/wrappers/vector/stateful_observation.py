"""A collection of stateful observation wrappers.

* ``NormalizeObservation`` - Normalize the observations
"""

from __future__ import annotations

from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.logger import warn
from gymnasium.vector.vector_env import (
    AutoresetMode,
    VectorEnv,
    VectorObservationWrapper,
)
from gymnasium.wrappers.utils import RunningMeanStd


__all__ = ["NormalizeObservation"]


class NormalizeObservation(VectorObservationWrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the observation
    statistics. If `True` (default), the `RunningMeanStd` will get updated every step and reset call.
    If `False`, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.

    Example without the normalize reward wrapper:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
        >>> obs, info = envs.reset(seed=123)
        >>> _ = envs.action_space.seed(123)
        >>> for _ in range(100):
        ...     obs, *_ = envs.step(envs.action_space.sample())
        >>> np.mean(obs)
        np.float32(0.024251968)
        >>> np.std(obs)
        np.float32(0.62259156)
        >>> envs.close()

    Example with the normalize reward wrapper:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
        >>> envs = NormalizeObservation(envs)
        >>> obs, info = envs.reset(seed=123)
        >>> _ = envs.action_space.seed(123)
        >>> for _ in range(100):
        ...     obs, *_ = envs.step(envs.action_space.sample())
        >>> np.mean(obs)
        np.float32(-0.2359734)
        >>> np.std(obs)
        np.float32(1.1938739)
        >>> envs.close()
    """

    def __init__(self, env: VectorEnv, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        VectorObservationWrapper.__init__(self, env)

        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` data. Assuming that the vector environment it follows the `NextStep` autoreset api or autoreset is disabled. Read https://farama.org/Vector-Autoreset-Mode for more details."
            )
        else:
            assert self.env.metadata["autoreset_mode"] in {AutoresetMode.NEXT_STEP}

        self.obs_rms = RunningMeanStd(
            shape=self.single_observation_space.shape,
            dtype=self.single_observation_space.dtype,
        )
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the observation statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset function for `NormalizeObservationWrapper` which is disabled for partial resets."""
        assert (
            options is None
            or "reset_mask" not in options
            or np.all(options["reset_mask"])
        )
        return super().reset(seed=seed, options=options)

    def observations(self, observations: ObsType) -> ObsType:
        """Defines the vector observation normalization function.

        Args:
            observations: A vector observation from the environment

        Returns:
            the normalized observation
        """
        if self._update_running_mean:
            self.obs_rms.update(observations)
        return (observations - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )
