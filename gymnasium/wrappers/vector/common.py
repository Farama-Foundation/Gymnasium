"""Wrapper that tracks the cumulative rewards and episode lengths."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Generic

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

from gymnasium.logger import warn
from gymnasium.typing import VectorActType, VectorObsType
from gymnasium.vector.vector_env import AutoresetMode, VectorEnv, VectorWrapper

__all__ = ["RecordEpisodeStatistics"]


# A specialised, `np.ndarray`-bound reward array type. This is *not* the shared
# `gymnasium.typing.RewardArrayType` (which is unbounded), so it keeps a distinct name.
NDRewardArrayType = TypeVar("NDRewardArrayType", bound=np.ndarray, default=Any)


class RecordEpisodeStatistics(
    VectorWrapper[
        VectorObsType, VectorActType, NDRewardArrayType, npt.NDArray[np.bool_]
    ],
    Generic[VectorObsType, VectorActType, NDRewardArrayType],
):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of any episode within the vectorized env, the statistics of the episode
    will be added to ``info`` using the key ``episode``, and the ``_episode`` key
    is used to indicate the environment index which has a terminated or truncated episode.

        >>> infos = {  # doctest: +SKIP
        ...     ...
        ...     "episode": {
        ...         "r": "<array of cumulative reward for each done sub-environment>",
        ...         "l": "<array of episode length for each done sub-environment>",
        ...         "t": "<array of elapsed time since beginning of episode for each done sub-environment>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes

    Example:
        >>> from pprint import pprint
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers.vector import RecordEpisodeStatistics
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3)
        >>> envs = RecordEpisodeStatistics(envs)
        >>> obs, info = envs.reset(123)
        >>> _ = envs.action_space.seed(123)
        >>> end = False
        >>> while not end:
        ...     obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
        ...     end = term.any() or trunc.any()
        ...
        >>> envs.close()
        >>> pprint(info) # doctest: +SKIP
        {'_episode': array([ True, False, False]),
         'episode': {'l': array([11,  0,  0], dtype=int32),
                     'r': array([11.,  0.,  0.], dtype=float32),
                     't': array([0.007812, 0.      , 0.      ], dtype=float32)},
    """

    episode_count: int

    # 1-d arrays
    episode_start_times: np.ndarray[tuple[int], np.dtype[np.float64]]
    episode_returns: np.ndarray[tuple[int], np.dtype[np.float64]]
    episode_lengths: np.ndarray[tuple[int], np.dtype[np.int_]]
    prev_dones: np.ndarray[tuple[int], np.dtype[np.bool_]]

    time_queue: deque[np.float64]
    return_queue: deque[np.float64]
    length_queue: deque[np.int_]

    def __init__(
        self,
        env: VectorEnv[
            VectorObsType, VectorActType, NDRewardArrayType, npt.NDArray[np.bool_]
        ],
        buffer_length: int = 100,
        stats_key: str = "episode",
    ) -> None:
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            buffer_length: The size of the buffers :attr:`return_queue`, :attr:`length_queue` and :attr:`time_queue`
            stats_key: The info key to save the data
        """
        super().__init__(env)
        self._stats_key = stats_key
        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` tag in its metadata, therefore, `RecordEpisodeStatistics` is assuming that the environment uses `AutoresetMode.NEXT_STEP`. See `https://farama.org/Vector-Autoreset-Mode` for more information on autoreset modes."
            )
            self._autoreset_mode = AutoresetMode.NEXT_STEP
        else:
            if not isinstance(self.env.metadata["autoreset_mode"], AutoresetMode):
                raise TypeError(
                    f"Expected env.metadata['autoreset_mode'] to be an AutoresetMode, got {type(self.env.metadata['autoreset_mode'])}"
                )
            self._autoreset_mode = self.env.metadata["autoreset_mode"]

        self.episode_count = 0

        self.episode_start_times = np.zeros((self.num_envs,))
        self.episode_returns = np.zeros((self.num_envs,))
        self.episode_lengths = np.zeros((self.num_envs,), dtype=int)
        self.prev_dones = np.zeros((self.num_envs,), dtype=bool)

        self.time_queue = deque(maxlen=buffer_length)
        self.return_queue = deque(maxlen=buffer_length)
        self.length_queue = deque(maxlen=buffer_length)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VectorObsType, dict[str, Any]]:
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(seed=seed, options=options)

        if options is not None and "reset_mask" in options:
            reset_mask = options.pop("reset_mask")
            if not isinstance(reset_mask, np.ndarray):
                raise TypeError(
                    f"`options['reset_mask']` must be a numpy array, got {type(reset_mask)}"
                )
            if reset_mask.shape != (self.num_envs,):
                raise ValueError(
                    f"`options['reset_mask']` must have shape `({self.num_envs},)`, got {reset_mask.shape}"
                )
            if reset_mask.dtype != np.bool_:
                raise TypeError(
                    f"`options['reset_mask']` must have `dtype=np.bool_`, got {reset_mask.dtype}"
                )
            if not np.any(reset_mask):
                raise ValueError(
                    f"`options['reset_mask']` must contain a boolean array with at least one True value, got reset_mask={reset_mask}"
                )

            self.episode_start_times[reset_mask] = time.perf_counter()
            self.episode_returns[reset_mask] = 0
            self.episode_lengths[reset_mask] = 0
            self.prev_dones[reset_mask] = False
        else:
            self.episode_start_times = np.full(self.num_envs, time.perf_counter())
            self.episode_returns = np.zeros(self.num_envs)
            self.episode_lengths = np.zeros(self.num_envs, dtype=int)
            self.prev_dones = np.zeros(self.num_envs, dtype=bool)

        return obs, info

    def step(
        self, actions: VectorActType
    ) -> tuple[
        VectorObsType,
        NDRewardArrayType,
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
        dict[str, Any],
    ]:
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        assert isinstance(infos, dict), (
            f"`vector.RecordEpisodeStatistics` requires `info` type to be `dict`, its actual type is {type(infos)}. This may be due to usage of other wrappers in the wrong order."
        )

        if self._autoreset_mode == AutoresetMode.SAME_STEP:
            # Sub-environments reset within the same step as they terminate or
            # truncate, therefore, every step counts towards an episode.
            self.episode_returns += rewards
            self.episode_lengths += 1
        else:
            # For `NEXT_STEP` autoreset, the step after a termination or
            # truncation resets the sub-environment and doesn't count towards
            # the next episode's statistics.
            self.episode_returns[self.prev_dones] = 0
            self.episode_returns[np.logical_not(self.prev_dones)] += rewards[
                np.logical_not(self.prev_dones)
            ]

            self.episode_lengths[self.prev_dones] = 0
            self.episode_lengths[~self.prev_dones] += 1

            self.episode_start_times[self.prev_dones] = time.perf_counter()

        self.prev_dones = dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)

        if num_dones:
            if self._stats_key in infos or f"_{self._stats_key}" in infos:
                raise ValueError(
                    f"Attempted to add episode stats with key '{self._stats_key}' but this key already exists in info: {list(infos.keys())}"
                )
            else:
                episode_time_length = np.round(
                    time.perf_counter() - self.episode_start_times, 6
                )
                infos[self._stats_key] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(dones, episode_time_length, 0.0),
                }
                infos[f"_{self._stats_key}"] = dones

            self.episode_count += num_dones
            for i in np.where(dones):
                self.time_queue.extend(episode_time_length[i])
                self.return_queue.extend(self.episode_returns[i])
                self.length_queue.extend(self.episode_lengths[i])

            if self._autoreset_mode == AutoresetMode.SAME_STEP:
                # The done sub-environments have already been reset, restart
                # their statistics immediately rather than on the next step.
                self.episode_returns[dones] = 0
                self.episode_lengths[dones] = 0
                self.episode_start_times[dones] = time.perf_counter()

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
