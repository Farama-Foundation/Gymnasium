"""A collection of stateful observation wrappers.

* ``FrameStackObservation`` - Frame stack the observations
* ``NormalizeObservation`` - Normalize the observations
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Final, cast

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.error import InvalidBound
from gymnasium.logger import warn
from gymnasium.spaces import Box
from gymnasium.vector.utils import (
    batch_space,
    concatenate,
    create_empty_array,
    iterate,
)
from gymnasium.vector.vector_env import (
    ArrayType,
    AutoresetMode,
    VectorEnv,
    VectorObservationWrapper,
    VectorWrapper,
)
from gymnasium.wrappers.utils import RunningMeanStd, create_zero_array

__all__ = ["FrameStackObservation", "NormalizeObservation"]


class FrameStackObservation(
    VectorWrapper[WrapperObsType, ActType, ObsType, ActType, ArrayType],
    gym.utils.RecordConstructorArgs,
):
    """Stacks the observations from the last ``N`` time steps in a rolling manner for each sub-environment.

    For example, if the number of stacks is 4 and the vector environment has 3 sub-environments
    with single observation shape ``(4,)``, then the returned observation has shape ``(3, 4, 4)``,
    and :attr:`single_observation_space` has shape ``(4, 4)``.

    Users have options for the padded observation used when a sub-environment resets:

     * ``"reset"`` (default) - The reset value is repeated
     * ``"zero"`` - A "zero"-like instance of the single observation space
     * custom - An instance of the single observation space

    Supports all :class:`gymnasium.vector.AutoresetMode` values (``NEXT_STEP``, ``SAME_STEP``,
    and ``DISABLED``), as well as partial resets via ``reset(options={"reset_mask": mask})``.
    For ``AutoresetMode.SAME_STEP``, ``info["final_obs"]`` is also stacked with the episode's
    terminal observation before the corresponding sub-environment's frame buffer is reset.

    A single-environment version of the wrapper exists :class:`gymnasium.wrappers.FrameStackObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers.vector import FrameStackObservation
        >>> envs = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")
        >>> envs = FrameStackObservation(envs, stack_size=3)
        >>> envs.single_observation_space
        Box([[-4.8               -inf -0.41887903        -inf]
         [-4.8               -inf -0.41887903        -inf]
         [-4.8               -inf -0.41887903        -inf]], [[4.8               inf 0.41887903        inf]
         [4.8               inf 0.41887903        inf]
         [4.8               inf 0.41887903        inf]], (3, 4), float32)
        >>> obs, _ = envs.reset(seed=123)
        >>> obs.shape
        (2, 3, 4)
        >>> envs.close()

    Example with zero padding:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers.vector import FrameStackObservation
        >>> envs = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")
        >>> envs = FrameStackObservation(envs, stack_size=3, padding_type="zero")
        >>> obs, _ = envs.reset(seed=123)
        >>> obs[0]
        array([[ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
              dtype=float32)
        >>> envs.close()
    """

    stack_size: Final[int]
    padding_type: Final[str]
    padding_value: Any
    _autoreset_mode: AutoresetMode

    def __init__(
        self,
        env: VectorEnv[ObsType, ActType, ArrayType],
        stack_size: int,
        *,
        padding_type: str | Any = "reset",
    ) -> None:
        """Vector observation wrapper that stacks the observations in a rolling manner.

        Args:
            env: The vector environment to apply the wrapper to.
            stack_size: The number of frames to stack for each sub-environment.
            padding_type: The padding type to use when stacking observations after a reset:
                ``"reset"``, ``"zero"``, or a custom observation in ``env.single_observation_space``.

        Raises:
            TypeError: If ``stack_size`` is not an integer or ``env.metadata["autoreset_mode"]`` is invalid.
            ValueError: If ``stack_size <= 0`` or ``padding_type`` is invalid.
        """
        if not np.issubdtype(type(stack_size), np.integer):
            raise TypeError(
                f"The stack_size is expected to be an integer, actual type: {type(stack_size)}"
            )
        if not 0 < stack_size:
            raise ValueError(
                f"The stack_size needs to be greater than zero, actual value: {stack_size}"
            )

        gym.utils.RecordConstructorArgs.__init__(
            self, stack_size=stack_size, padding_type=padding_type
        )
        VectorWrapper.__init__(self, env)

        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` tag in its metadata, therefore, `FrameStackObservation` is assuming that the environment uses `AutoresetMode.NEXT_STEP`. See `https://farama.org/Vector-Autoreset-Mode` for more information on autoreset modes."
            )
            self._autoreset_mode = AutoresetMode.NEXT_STEP
        else:
            if not isinstance(self.env.metadata["autoreset_mode"], AutoresetMode):
                raise TypeError(
                    f"Expected env.metadata['autoreset_mode'] to be an AutoresetMode, got {type(self.env.metadata['autoreset_mode'])}"
                )
            self._autoreset_mode = self.env.metadata["autoreset_mode"]

        if isinstance(padding_type, str) and (
            padding_type == "reset" or padding_type == "zero"
        ):
            self.padding_value = create_zero_array(env.single_observation_space)
        elif padding_type in env.single_observation_space:
            self.padding_value = deepcopy(padding_type)
            padding_type = "_custom"
        else:
            if isinstance(padding_type, str):
                raise ValueError(
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r}"
                )
            else:
                raise ValueError(
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r} not an instance of env observation ({env.single_observation_space})"
                )

        self.stack_size: Final[int] = int(stack_size)
        self.padding_type: Final[str] = padding_type

        new_single_space = batch_space(env.single_observation_space, n=self.stack_size)
        self.single_observation_space = new_single_space
        self.observation_space = batch_space(new_single_space, n=self.num_envs)

        self._obs_queues: list[deque[Any]] = [
            deque(
                [deepcopy(self.padding_value) for _ in range(self.stack_size)],
                maxlen=self.stack_size,
            )
            for _ in range(self.num_envs)
        ]
        self._single_stacked_obs = create_empty_array(
            env.single_observation_space, n=self.stack_size
        )
        self._batched_stacked_obs = create_empty_array(
            self.single_observation_space, n=self.num_envs
        )
        self._prev_dones = np.zeros((self.num_envs,), dtype=np.bool_)

    def _reset_env_queue(self, env_idx: int, obs: Any) -> None:
        """Reset the frame queue for a single sub-environment using its reset observation."""
        pad_value = obs if self.padding_type == "reset" else self.padding_value
        queue = self._obs_queues[env_idx]
        for _ in range(self.stack_size - 1):
            queue.append(deepcopy(pad_value))
        queue.append(deepcopy(obs))

    def _stack_single_env(self, env_idx: int) -> Any:
        """Build a stacked observation copy for a single sub-environment."""
        return deepcopy(
            concatenate(
                self.env.single_observation_space,
                self._obs_queues[env_idx],
                self._single_stacked_obs,
            )
        )

    def _stack_all_envs(self) -> WrapperObsType:
        """Build the batched stacked observation copy across all sub-environments."""
        per_env_stacked = tuple(self._stack_single_env(i) for i in range(self.num_envs))
        updated_obs = deepcopy(
            concatenate(
                self.single_observation_space,
                per_env_stacked,
                self._batched_stacked_obs,
            )
        )
        return cast("WrapperObsType", updated_obs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Reset the vector environment and update frame buffers for reset sub-environments.

        Args:
            seed: The seed to reset the sub-environments.
            options: Optional reset options, including ``reset_mask`` for partial resets.

        Returns:
            The batched stacked observations and infos.
        """
        reset_mask = None if options is None else options.get("reset_mask")
        observations, infos = self.env.reset(seed=seed, options=options)

        per_env_obs = tuple(iterate(self.env.observation_space, observations))
        if reset_mask is not None:
            mask = np.asarray(reset_mask, dtype=np.bool_)
            for i in range(self.num_envs):
                if mask[i]:
                    self._reset_env_queue(i, per_env_obs[i])
                    self._prev_dones[i] = False
        else:
            for i in range(self.num_envs):
                self._reset_env_queue(i, per_env_obs[i])
            self._prev_dones = np.zeros((self.num_envs,), dtype=np.bool_)

        return self._stack_all_envs(), infos

    def step(
        self, actions: ActType
    ) -> tuple[WrapperObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Step through the vector environment, updating each sub-environment's frame buffer.

        Args:
            actions: The batched actions to step through the vector environment with.

        Returns:
            Batched stacked observations, rewards, terminations, truncations, and infos.
        """
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        per_env_obs = tuple(iterate(self.env.observation_space, observations))
        dones = np.logical_or(terminations, truncations)

        if self._autoreset_mode == AutoresetMode.SAME_STEP:
            if "final_obs" in infos and "_final_obs" in infos:
                final_obs_mask = np.asarray(infos["_final_obs"], dtype=np.bool_)
                final_obs = infos["final_obs"].copy()
                for i in range(self.num_envs):
                    if final_obs_mask[i]:
                        self._obs_queues[i].append(deepcopy(final_obs[i]))
                        final_obs[i] = self._stack_single_env(i)
                        self._reset_env_queue(i, per_env_obs[i])
                    elif dones[i]:
                        self._reset_env_queue(i, per_env_obs[i])
                    else:
                        self._obs_queues[i].append(deepcopy(per_env_obs[i]))
                infos["final_obs"] = final_obs
            else:
                for i in range(self.num_envs):
                    if dones[i]:
                        self._reset_env_queue(i, per_env_obs[i])
                    else:
                        self._obs_queues[i].append(deepcopy(per_env_obs[i]))
            self._prev_dones = np.zeros((self.num_envs,), dtype=np.bool_)
        elif self._autoreset_mode == AutoresetMode.NEXT_STEP:
            for i in range(self.num_envs):
                if self._prev_dones[i]:
                    self._reset_env_queue(i, per_env_obs[i])
                else:
                    self._obs_queues[i].append(deepcopy(per_env_obs[i]))
            self._prev_dones = np.asarray(dones, dtype=np.bool_).copy()
        else:
            for i in range(self.num_envs):
                self._obs_queues[i].append(deepcopy(per_env_obs[i]))

        return self._stack_all_envs(), rewards, terminations, truncations, infos


class NormalizeObservation(
    VectorObservationWrapper[np.ndarray, ActType, np.ndarray, ArrayType],
    gym.utils.RecordConstructorArgs,
):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the observation
    statistics. If `True` (default), the `RunningMeanStd` will get updated every step and reset call.
    If `False`, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.

    Note:
        For same-step autoreset, ``info["final_obs"]`` is normalized and included in the statistics before the
        step's observations, matching the update order of :class:`gymnasium.wrappers.NormalizeObservation`.
        Partial resets (``reset_mask`` with any ``False`` values) are not supported for disabled autoreset.

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

    autoreset_mode: AutoresetMode

    def __init__(
        self, env: VectorEnv[np.ndarray, ActType, ArrayType], epsilon: float = 1e-8
    ) -> None:
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.

        Raises:
            InvalidBound: If ``epsilon`` is not finite and strictly positive.
        """
        # `epsilon` is added under the square root to keep the division away from
        # zero, so a non-positive value defeats its purpose and, once it exceeds
        # the variance, silently turns every normalized observation into NaN.
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise InvalidBound(
                f"`epsilon` should be strictly positive and finite. Received {epsilon}"
            )

        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        # Skips `VectorObservationWrapper.__init__` as it rejects same-step autoreset, supported by `step` normalizing `info["final_obs"]`
        VectorWrapper.__init__(self, env)

        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` data. Assuming that the vector environment it follows the `NextStep` autoreset api or autoreset is disabled. Read https://farama.org/Vector-Autoreset-Mode for more details."
            )
            self.autoreset_mode = AutoresetMode.NEXT_STEP
        else:
            if not isinstance(self.env.metadata["autoreset_mode"], AutoresetMode):
                raise TypeError(
                    f"Expected env.metadata['autoreset_mode'] to be an AutoresetMode, got {type(self.env.metadata['autoreset_mode'])}"
                )
            self.autoreset_mode = self.env.metadata["autoreset_mode"]

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
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset function for `NormalizeObservationWrapper` which is disabled for partial resets."""
        if options is not None and "reset_mask" in options:
            if not np.all(options["reset_mask"]):
                raise ValueError(
                    "NormalizeObservation does not support partial resets. The 'reset_mask' must contain all True values."
                )
        return super().reset(seed=seed, options=options)

    def step(
        self, actions: ActType
    ) -> tuple[np.ndarray, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through the environment, for same-step autoreset normalizing ``info["final_obs"]`` before the observations."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        if self.autoreset_mode == AutoresetMode.SAME_STEP and "final_obs" in infos:
            # The final observations occur before the reset observations within the step, therefore, are normalized first
            final_obs_mask = infos["_final_obs"]
            final_obs = infos["final_obs"]
            normalized_final_obs = self.observations(
                np.stack(final_obs[final_obs_mask])
            )
            for i, normalized_obs in zip(
                np.flatnonzero(final_obs_mask), normalized_final_obs, strict=True
            ):
                final_obs[i] = normalized_obs

        return (
            self.observations(observations),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def observations(
        self, observations: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
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
