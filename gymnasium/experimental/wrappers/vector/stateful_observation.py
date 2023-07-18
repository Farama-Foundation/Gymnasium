"""A collection of stateful observation wrappers.

* ``NormalizeObservationV0`` - Normalize the observations
"""
from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.experimental.vector.vector_env import VectorObservationWrapper
from gymnasium.experimental.wrappers.utils import RunningMeanStd


__all__ = ["NormalizeObservationV0"]


class NormalizeObservationV0(VectorObservationWrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the observation
    statistics. If `True` (default), the `RunningMeanStd` will get updated every step and reset call.
    If `False`, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env[ObsType, ActType], epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.ObservationWrapper.__init__(self, env)

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

    def vector_observation(self, observation: ObsType) -> ObsType:
        """Defines the vector observation normalization function.

        Args:
            observation: A vector observation from the environment

        Returns:
            the normalized observation
        """
        return self._normalize_observations(observation)

    def single_observation(self, observation: ObsType) -> ObsType:
        """Defines the single observation normalization function.

        Args:
            observation: A single observation from the environment

        Returns:
            The normalized observation
        """
        return self._normalize_observations(observation[None])

    def _normalize_observations(self, observations: ObsType) -> ObsType:
        if self._update_running_mean:
            self.obs_rms.update(observations)
        return (observations - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )
