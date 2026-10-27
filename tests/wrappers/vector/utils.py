"""Utility functions for testing the vector wrappers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import (
    AsyncVectorEnv,
    AutoresetMode,
    SyncVectorEnv,
    VectorEnv,
    VectorObservationWrapper,
    VectorWrapper,
)


def staggered_episode_lengths(num_envs: int, min_episode_steps: int = 5) -> list[int]:
    """Returns a different maximum episode length for each sub-environment, so episodes end on different steps."""
    return [min_episode_steps + 2 * i for i in range(num_envs)]


def _make_env(
    env_id: str | Callable[[], gym.Env],
    max_episode_steps: int,
    wrappers: Sequence[Callable[[gym.Env], gym.Env]],
) -> gym.Env:
    if isinstance(env_id, str):
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
    else:
        env = gym.wrappers.TimeLimit(env_id(), max_episode_steps=max_episode_steps)
    for wrapper in wrappers:
        env = wrapper(env)
    return env


def make_staggered_vector_env(
    env_id: str | Callable[[], gym.Env],
    num_envs: int,
    autoreset_mode: AutoresetMode,
    wrappers: Sequence[Callable[[gym.Env], gym.Env]] = (),
    episode_lengths: Sequence[int] | None = None,
    vectorization_mode: str = "sync",
) -> VectorEnv:
    """Creates a vector environment where each sub-environment has a different maximum episode length.

    Unlike ``gym.make_vec(..., max_episode_steps=n)``, the sub-environments don't end their episodes
    in lockstep, therefore, the autoreset (or ``reset_mask``) code paths are exercised with partial masks.
    """
    if episode_lengths is None:
        episode_lengths = staggered_episode_lengths(num_envs)
    assert len(episode_lengths) == num_envs

    env_fns = [
        partial(_make_env, env_id, max_episode_steps, wrappers)
        for max_episode_steps in episode_lengths
    ]
    if vectorization_mode == "sync":
        return SyncVectorEnv(env_fns, autoreset_mode=autoreset_mode)
    elif vectorization_mode == "async":
        return AsyncVectorEnv(env_fns, autoreset_mode=autoreset_mode)
    raise ValueError(f"Unknown vectorization mode: {vectorization_mode}")


class EpisodeEndTracker:
    """Records the sub-environment episode ends, to check the autoreset code paths were exercised.

    For multiple sub-environments, this checks that some steps only had a subset of the sub-environments end,
    otherwise masks over sub-environments (``_final_obs``, ``reset_mask``, etc.) are only tested as all ``True``.
    """

    def __init__(self, num_envs: int, episode_lengths: Sequence[int] | None = None):
        """Initialises the tracker, ``episode_lengths`` defaults to :func:`staggered_episode_lengths`."""
        self.num_envs = num_envs
        self.episode_lengths = (
            staggered_episode_lengths(num_envs)
            if episode_lengths is None
            else episode_lengths
        )
        self.num_episode_ends = np.zeros(num_envs, dtype=np.int64)
        self.num_partial_steps = 0

    def update(self, terminations: np.ndarray, truncations: np.ndarray) -> np.ndarray:
        """Records the episode ends of a step and returns which sub-environments ended."""
        dones = np.logical_or(terminations, truncations)
        self.num_episode_ends += dones
        if np.any(dones) and not np.all(dones):
            self.num_partial_steps += 1
        return dones

    def check(self, num_steps: int):
        """Checks every sub-environment ended its episode multiple times and, for multiple sub-environments, not in lockstep."""
        # `episode_length + 1` as next-step autoreset uses an additional step to reset,
        # episodes can terminate before being truncated, therefore, this is a lower bound
        for i, episode_length in enumerate(self.episode_lengths):
            min_episode_ends = num_steps // (episode_length + 1)
            assert min_episode_ends >= 2, (
                f"`num_steps` ({num_steps}) is too small for sub-environment {i} to end its episode multiple times"
            )
            assert self.num_episode_ends[i] >= min_episode_ends, (
                f"sub-environment {i} episode ends: {self.num_episode_ends[i]}, expected at least {min_episode_ends}"
            )

        if self.num_envs > 1:
            assert self.num_partial_steps > 0, (
                "Sub-environment episodes only ended in lockstep"
            )


def check_vector_wrapper_equivalence(
    autoreset_mode: AutoresetMode,
    num_envs: int,
    env_id: str,
    vector_wrapper: type[VectorWrapper],
    vector_kwargs: dict[str, Any],
    env_wrapper: type[gym.Wrapper],
    env_kwargs: dict[str, Any],
    vectorization_mode: str = "sync",
    num_steps: int = 50,
):
    """Checks that `vector_wrapper(VectorEnv)` and `VectorEnv(env_wrapper)` are equivalent.

    The sub-environments have short and different episode lengths (see :func:`make_staggered_vector_env`)
    so every sub-environment is reset (via autoreset or `reset_mask`) multiple times within `num_steps`,
    including steps where only some sub-environments are reset.

    Additionally, checks that `vector_wrapper` doesn't modify the actions in-place.
    """
    if (
        autoreset_mode == AutoresetMode.SAME_STEP
        and issubclass(vector_wrapper, VectorObservationWrapper)
        and not issubclass(
            vector_wrapper,
            (
                gym.wrappers.vector.VectorizeTransformObservation,
                gym.wrappers.vector.NormalizeObservation,
            ),
        )
    ):
        # Vector observation wrappers that don't transform `info["final_obs"]` reject same-step autoreset
        with pytest.raises(ValueError, match="Expected autoreset_mode to be"):
            vector_wrapper(
                gym.make_vec(
                    id=env_id,
                    num_envs=num_envs,
                    vectorization_mode=vectorization_mode,
                    vector_kwargs={"autoreset_mode": autoreset_mode},
                ),
                **vector_kwargs,
            )
        return

    wrapper_vector_env: VectorEnv = vector_wrapper(
        make_staggered_vector_env(
            env_id, num_envs, autoreset_mode, vectorization_mode=vectorization_mode
        ),
        **vector_kwargs,
    )
    vector_wrapper_env = make_staggered_vector_env(
        env_id,
        num_envs,
        autoreset_mode,
        wrappers=(lambda env: env_wrapper(env, **env_kwargs),),
        vectorization_mode=vectorization_mode,
    )

    assert wrapper_vector_env.metadata["autoreset_mode"] == autoreset_mode
    assert vector_wrapper_env.metadata["autoreset_mode"] == autoreset_mode

    assert wrapper_vector_env.action_space == vector_wrapper_env.action_space
    assert wrapper_vector_env.observation_space == vector_wrapper_env.observation_space
    assert (
        wrapper_vector_env.single_action_space == vector_wrapper_env.single_action_space
    )
    assert (
        wrapper_vector_env.single_observation_space
        == vector_wrapper_env.single_observation_space
    )

    assert wrapper_vector_env.num_envs == vector_wrapper_env.num_envs

    wrapper_vector_obs, wrapper_vector_info = wrapper_vector_env.reset(seed=123)
    vector_wrapper_obs, vector_wrapper_info = vector_wrapper_env.reset(seed=123)

    assert data_equivalence(wrapper_vector_obs, vector_wrapper_obs)
    assert data_equivalence(wrapper_vector_info, vector_wrapper_info)

    wrapper_vector_env.action_space.seed(123)
    tracker = EpisodeEndTracker(num_envs)
    for _ in range(num_steps):
        action = wrapper_vector_env.action_space.sample()
        # Each environment is given its own copy of the action, so an in-place modification is caught below
        # rather than being passed to the other environment
        action_copy = deepcopy(action)
        wrapper_vector_step_returns = wrapper_vector_env.step(action)
        assert data_equivalence(action, action_copy), (
            f"{vector_wrapper} modified the action in-place"
        )
        vector_wrapper_step_returns = vector_wrapper_env.step(action_copy)

        for wrapper_vector_return, vector_wrapper_return in zip(
            wrapper_vector_step_returns, vector_wrapper_step_returns, strict=True
        ):
            assert data_equivalence(wrapper_vector_return, vector_wrapper_return)

        _, _, terminations, truncations, _ = wrapper_vector_step_returns
        episode_ends = tracker.update(terminations, truncations)

        if autoreset_mode == AutoresetMode.DISABLED and np.any(episode_ends):
            wrapper_vector_obs, wrapper_vector_info = wrapper_vector_env.reset(
                options={"reset_mask": episode_ends}
            )
            vector_wrapper_obs, vector_wrapper_info = vector_wrapper_env.reset(
                options={"reset_mask": episode_ends}
            )

            assert data_equivalence(wrapper_vector_obs, vector_wrapper_obs)
            assert data_equivalence(wrapper_vector_info, vector_wrapper_info)

    # Ensures that the autoreset (or `reset_mask`) code path was exercised for every sub-environment
    tracker.check(num_steps)

    wrapper_vector_env.close()
    vector_wrapper_env.close()
