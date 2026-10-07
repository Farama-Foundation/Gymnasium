"""Tests that the vectorised wrappers operate identically in `VectorEnv(Wrapper)` and `VectorWrapper(VectorEnv)`.

The exception is the data converter wrappers
 * Data conversion wrappers - `JaxToTorch`, `JaxToNumpy` and `NumpyToJax`
 * Normalizing wrappers - `NormalizeObservation` and `NormalizeReward`
 * Different implementations - `LambdaObservation`, `LambdaReward` and `LambdaAction`
 * Different random sources - `StickyAction`
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import wrappers
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorWrapper
from gymnasium.vector.vector_env import AutoresetMode
from tests.testing_env import GenericTestEnv


@pytest.fixture
def custom_environments():
    gym.register(
        "DictObsEnv-v0",
        lambda: GenericTestEnv(
            observation_space=Dict({"a": Box(0, 1), "b": Discrete(5)})
        ),
    )

    yield

    del gym.registry["DictObsEnv-v0"]


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
@pytest.mark.parametrize("num_envs", (1, 3))
@pytest.mark.parametrize(
    "env_id, wrapper_name, kwargs",
    (
        ("DictObsEnv-v0", "FilterObservation", {"filter_keys": ["a"]}),
        ("CartPole-v1", "FlattenObservation", {}),
        ("CarRacing-v3", "GrayscaleObservation", {}),
        ("CarRacing-v3", "ResizeObservation", {"shape": (35, 45)}),
        ("CarRacing-v3", "ReshapeObservation", {"shape": (96, 48, 6)}),
        (
            "CartPole-v1",
            "RescaleObservation",
            {
                "min_obs": np.array([0, -np.inf, 0, -np.inf]),
                "max_obs": np.array([1, np.inf, 1, np.inf]),
            },
        ),
        ("CarRacing-v3", "DtypeObservation", {"dtype": np.int32}),
        # ("CartPole-v1", "RenderObservation", {}),  # not implemented
        # ("CartPole-v1", "TimeAwareObservation", {}),  # not implemented
        # ("CartPole-v1", "FrameStackObservation", {}),  # not implemented
        # ("CartPole-v1", "DelayObservation", {}),  # not implemented
        ("MountainCarContinuous-v0", "ClipAction", {}),
        (
            "MountainCarContinuous-v0",
            "RescaleAction",
            {"min_action": 1, "max_action": 2},
        ),
        ("CartPole-v1", "ClipReward", {"min_reward": -0.25, "max_reward": 0.75}),
    ),
)
def test_vector_wrapper_equivalence(
    autoreset_mode: AutoresetMode,
    num_envs: int,
    env_id: str,
    wrapper_name: str,
    kwargs: dict[str, Any],
    custom_environments,  # pytest fixture
):
    """Checks that `VectorWrapper(VectorEnv)` and `VectorEnv(Wrapper)` are equivalent for every autoreset mode."""
    check_vector_wrapper_equivalence(
        autoreset_mode,
        num_envs,
        env_id,
        getattr(wrappers.vector, wrapper_name),
        kwargs,
        getattr(wrappers, wrapper_name),
        kwargs,
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
    max_episode_steps: int = 7,
):
    """Checks that `vector_wrapper(VectorEnv)` and `VectorEnv(env_wrapper)` are equivalent.

    `max_episode_steps` is kept small so every sub-environment is reset (via autoreset or `reset_mask`)
    multiple times within `num_steps`, exercising the autoreset code paths of each mode.
    """
    if (
        autoreset_mode == AutoresetMode.SAME_STEP
        and issubclass(vector_wrapper, VectorObservationWrapper)
        and not vector_wrapper.supports_same_step_autoreset
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
        gym.make_vec(
            id=env_id,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            vector_kwargs={"autoreset_mode": autoreset_mode},
            max_episode_steps=max_episode_steps,
        ),
        **vector_kwargs,
    )
    vector_wrapper_env = gym.make_vec(
        id=env_id,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        vector_kwargs={"autoreset_mode": autoreset_mode},
        wrappers=(lambda env: env_wrapper(env, **env_kwargs),),
        max_episode_steps=max_episode_steps,
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
    num_episode_ends = 0
    for _ in range(num_steps):
        action = wrapper_vector_env.action_space.sample()
        wrapper_vector_step_returns = wrapper_vector_env.step(action)
        vector_wrapper_step_returns = vector_wrapper_env.step(action)

        for wrapper_vector_return, vector_wrapper_return in zip(
            wrapper_vector_step_returns, vector_wrapper_step_returns, strict=True
        ):
            assert data_equivalence(wrapper_vector_return, vector_wrapper_return)

        _, _, terminations, truncations, _ = wrapper_vector_step_returns
        episode_ends = np.logical_or(terminations, truncations)
        num_episode_ends += int(np.sum(episode_ends))

        if autoreset_mode == AutoresetMode.DISABLED and np.any(episode_ends):
            wrapper_vector_obs, wrapper_vector_info = wrapper_vector_env.reset(
                options={"reset_mask": episode_ends}
            )
            vector_wrapper_obs, vector_wrapper_info = vector_wrapper_env.reset(
                options={"reset_mask": episode_ends}
            )

            assert data_equivalence(wrapper_vector_obs, vector_wrapper_obs)
            assert data_equivalence(wrapper_vector_info, vector_wrapper_info)

    # Ensures that the autoreset (or `reset_mask`) code path was actually exercised for every sub-environment,
    # `max_episode_steps + 1` as next-step autoreset uses an additional step to reset
    assert num_episode_ends >= num_envs * (num_steps // (max_episode_steps + 1))

    wrapper_vector_env.close()
    vector_wrapper_env.close()
