"""A collection of stateful observation wrappers.

* ``NormalizeObservation`` - Normalize the observations
"""

from __future__ import annotations

from typing import Any, Generic, TypeAlias

import numpy as np
import numpy.typing as npt

import gymnasium as gym
from gymnasium.logger import warn
from gymnasium.spaces import Box
from gymnasium.typing import VectorActType, VectorBoolType, VectorRewardType
from gymnasium.vector.utils import batch_space
from gymnasium.vector.vector_env import (
    AutoresetMode,
    VectorEnv,
    VectorObservationWrapper,
)
from gymnasium.wrappers.utils import RunningMeanStd

__all__ = ["NormalizeObservation"]


# Shape-generic: the batched vector observation is (num_envs, *obs_shape), so
# the rank depends on the sub-environment's observation (2-D for CartPole,
# higher for images).
VectorFloat32Array: TypeAlias = npt.NDArray[np.float32]
VectorFloatingArray: TypeAlias = npt.NDArray[np.floating]


class NormalizeObservation(
    VectorObservationWrapper[
        VectorFloat32Array,
        VectorActType,
        VectorRewardType,
        VectorBoolType,
        VectorFloatingArray,
    ],
    gym.utils.RecordConstructorArgs,
    Generic[VectorActType, VectorRewardType, VectorBoolType],
):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the observation
    statistics. If `True` (default), the `RunningMeanStd` will get updated every step and reset call.
    If `False`, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.

    Example without the normalize observation wrapper:
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

    Example with the normalize observation wrapper:
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

    single_observation_space: Box  # f32
    observation_space: Box  # f32
    obs_rms: RunningMeanStd  # f32
    epsilon: float
    _update_running_mean: bool

    def __init__(
        self,
        env: VectorEnv[
            VectorFloatingArray, VectorActType, VectorRewardType, VectorBoolType
        ],
        epsilon: float = 1e-8,
    ) -> None:
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
            if self.env.metadata["autoreset_mode"] not in {AutoresetMode.NEXT_STEP}:
                raise ValueError(
                    f"Expected env.metadata['autoreset_mode'] to be AutoresetMode.NEXT_STEP, got {self.env.metadata['autoreset_mode']}"
                )

        new_single_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=self.single_observation_space.shape,
            dtype=np.float32,
        )
        self.single_observation_space = new_single_space
        # TODO: remove ignore comment once `ty` supports `@single_dispatch`
        self.observation_space = batch_space(new_single_space, self.num_envs)  # ty:ignore[invalid-assignment]

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
    def update_running_mean(self, setting: bool) -> None:
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VectorFloat32Array, dict[str, Any]]:
        """Reset function for `NormalizeObservationWrapper` which is disabled for partial resets."""
        if options is not None and "reset_mask" in options:
            if not np.all(options["reset_mask"]):
                raise ValueError(
                    "NormalizeObservation does not support partial resets. The 'reset_mask' must contain all True values."
                )
        return super().reset(seed=seed, options=options)

    def observations(self, observations: VectorFloatingArray) -> VectorFloat32Array:
        """Defines the vector observation normalization function.

        Args:
            observations: A vector observation from the environment

        Returns:
            the normalized observation
        """
        if self._update_running_mean:
            self.obs_rms.update(observations)
        return (
            (observations - self.obs_rms.mean)
            / np.sqrt(self.obs_rms.var + self.epsilon)
        ).astype(np.float32)
