"""A collection of stateful observation wrappers.

* ``NormalizeObservation`` - Normalize the observations
* ``TimeAwareObservation`` - Add per-environment elapsed episode time
"""

from __future__ import annotations

from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium.error import InvalidBound
from gymnasium.logger import warn
from gymnasium.spaces import Box
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array, iterate
from gymnasium.vector.vector_env import (
    AutoresetMode,
    VectorEnv,
    VectorObservationWrapper,
    VectorWrapper,
)
from gymnasium.wrappers.utils import RunningMeanStd

__all__ = ["NormalizeObservation", "TimeAwareObservation"]


class NormalizeObservation(VectorObservationWrapper, gym.utils.RecordConstructorArgs):
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

    def __init__(self, env: VectorEnv, epsilon: float = 1e-8) -> None:
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.

        Raises:
            InvalidBound: If ``epsilon`` is not strictly positive.
        """
        # `epsilon` is added under the square root to keep the division away from
        # zero, so a non-positive value defeats its purpose and, once it exceeds
        # the variance, silently turns every normalized observation into NaN.
        if epsilon <= 0:
            raise InvalidBound(
                f"`epsilon` should be strictly positive. Received {epsilon}"
            )

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
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset function for `NormalizeObservationWrapper` which is disabled for partial resets."""
        if options is not None and "reset_mask" in options:
            if not np.all(options["reset_mask"]):
                raise ValueError(
                    "NormalizeObservation does not support partial resets. The 'reset_mask' must contain all True values."
                )
        return super().reset(seed=seed, options=options)

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


class TimeAwareObservation(VectorWrapper, gym.utils.RecordConstructorArgs):
    """Add each environment's elapsed episode time to its observation.

    Matches :class:`gymnasium.wrappers.TimeAwareObservation`: time is an int32
    elapsed step count, or float32 elapsed time divided by the episode limit.
    Dict observations gain ``dict_time_key``, Tuple observations gain a final
    element, and other observations are placed under ``obs`` alongside ``time``.
    Flattening follows :func:`gymnasium.spaces.flatten` for each environment.

    Supports SyncVectorEnv and AsyncVectorEnv (including vector wrappers around
    them) with a common time limit and identical observation spaces. Limits are
    read from sub-environment specs, falling back to ``_max_episode_steps`` when
    all sub-environments expose it through their TimeLimit wrappers.

    All autoreset modes and partial resets are supported. SAME_STEP final
    observations contain terminal time; the returned reset observations have
    time zero. NEXT_STEP resets time on the subsequent reset-only step.

    Example:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")
        >>> envs = TimeAwareObservation(envs, flatten=False)
        >>> envs.reset(seed=42)[0]["time"]
        array([[0],
               [0]], dtype=int32)
        >>> envs.step([0, 1])[0]["time"]
        array([[1],
               [1]], dtype=int32)
        >>> envs.close()
    """

    def __init__(
        self,
        env: VectorEnv,
        flatten: bool = True,
        normalize_time: bool = False,
        *,
        dict_time_key: str = "time",
    ) -> None:
        """Initialize the per-environment clocks and augmented spaces.

        Args:
            env: The vector environment with a common finite episode limit.
            flatten: Flatten each augmented observation.
            normalize_time: Divide elapsed steps by the episode limit.
            dict_time_key: Time key for Dict observation spaces.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            flatten=flatten,
            normalize_time=normalize_time,
            dict_time_key=dict_time_key,
        )
        VectorWrapper.__init__(self, env)
        if "autoreset_mode" not in env.metadata:
            warn(f"{env} is missing `autoreset_mode` metadata. Assuming NEXT_STEP.")
            self.autoreset_mode = AutoresetMode.NEXT_STEP
        else:
            self.autoreset_mode = env.metadata["autoreset_mode"]
            if not isinstance(self.autoreset_mode, AutoresetMode):
                raise TypeError(
                    "Expected env.metadata['autoreset_mode'] to be an AutoresetMode, "
                    f"got {type(self.autoreset_mode)}"
                )

        base_env = env.unwrapped
        if not isinstance(
            base_env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)
        ):
            raise TypeError(
                "TimeAwareObservation requires SyncVectorEnv or AsyncVectorEnv."
            )
        if env.observation_space != batch_space(
            env.single_observation_space, env.num_envs
        ):
            raise ValueError(
                "TimeAwareObservation requires identical observation spaces."
            )
        limits = [
            spec.max_episode_steps if spec is not None else None
            for spec in base_env.get_attr("spec")
        ]
        if any(limit is None for limit in limits):
            # Check first: a failed attribute call would shut down async workers.
            if not all(base_env.call("has_wrapper_attr", "_max_episode_steps")):
                raise ValueError(
                    "Each environment must have a spec with max_episode_steps, "
                    "or all environments must expose a TimeLimit's _max_episode_steps."
                )
            fallback_limits = base_env.get_attr("_max_episode_steps")
            limits = [
                fallback if limit is None else limit
                for limit, fallback in zip(limits, fallback_limits, strict=True)
            ]
        if any(limit != limits[0] for limit in limits):
            raise ValueError(
                "TimeAwareObservation requires a common episode time limit."
            )
        limit = limits[0]
        if not isinstance(limit, (int, np.integer)) or limit <= 0:
            raise ValueError("The episode time limit must be a positive integer.")
        self.max_timesteps = int(limit)
        self.flatten = flatten
        self.normalize_time = normalize_time
        self.dict_time_key = dict_time_key
        self.timesteps = np.zeros(self.num_envs, dtype=np.int64)
        self._prev_dones = np.zeros(self.num_envs, dtype=np.bool_)

        time_space = (
            Box(0.0, 1.0)
            if normalize_time
            else Box(0, self.max_timesteps, dtype=np.int32)
        )
        space = env.single_observation_space
        if isinstance(space, gym.spaces.Dict):
            if dict_time_key in space.spaces:
                raise ValueError(
                    f"The `dict_time_key` ({dict_time_key!r}) already exists in the observation space."
                )
            self._time_observation_space = gym.spaces.Dict(
                {dict_time_key: time_space, **space.spaces}, sort_keys=space.sort_keys
            )
        elif isinstance(space, gym.spaces.Tuple):
            self._time_observation_space = gym.spaces.Tuple(
                space.spaces + (time_space,)
            )
        else:
            self._time_observation_space = gym.spaces.Dict(obs=space, time=time_space)
        self.single_observation_space = (
            gym.spaces.flatten_space(self._time_observation_space)
            if flatten
            else self._time_observation_space
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

    def _observation(self, observation: Any, timestep: int) -> Any:
        """Add time to one observation using the scalar wrapper's representation."""
        time = (
            np.array([timestep / self.max_timesteps], dtype=np.float32)
            if self.normalize_time
            else np.array([timestep], dtype=np.int32)
        )
        if isinstance(self.env.single_observation_space, gym.spaces.Dict):
            observation = {self.dict_time_key: time, **observation}
        elif isinstance(self.env.single_observation_space, gym.spaces.Tuple):
            observation = observation + (time,)
        else:
            observation = {"obs": observation, "time": time}
        if self.flatten:
            return gym.spaces.flatten(self._time_observation_space, observation)
        return observation

    def observations(self, observations: Any) -> Any:
        """Augment and batch observations without retaining a mutable output buffer."""
        return concatenate(
            self.single_observation_space,
            [
                self._observation(obs, int(t))
                for obs, t in zip(
                    iterate(self.env.observation_space, observations),
                    self.timesteps,
                    strict=True,
                )
            ],
            create_empty_array(self.single_observation_space, self.num_envs),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset selected clocks only after the underlying reset succeeds."""
        # Sync/AsyncVectorEnv pop reset_mask from options, so preserve it first.
        reset_mask = (
            options.get("reset_mask", slice(None))
            if options is not None
            else slice(None)
        )
        obs, info = self.env.reset(seed=seed, options=options)
        self.timesteps[reset_mask] = 0
        self._prev_dones[reset_mask] = False
        return self.observations(obs), info

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Advance clocks, transform terminal observations, and account for autoresets."""
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.timesteps += 1
        if self.autoreset_mode == AutoresetMode.NEXT_STEP:
            self.timesteps[self._prev_dones] = 0
        dones = np.logical_or(terminations, truncations)
        if self.autoreset_mode == AutoresetMode.SAME_STEP:
            if "final_obs" in infos:
                infos = infos.copy()
                final_obs = infos["final_obs"].copy()
                for i in np.flatnonzero(infos["_final_obs"]):
                    final_obs[i] = self._observation(
                        final_obs[i], int(self.timesteps[i])
                    )
                infos["final_obs"] = final_obs
            self.timesteps[dones] = 0
        self._prev_dones = dones
        return self.observations(obs), rewards, terminations, truncations, infos
