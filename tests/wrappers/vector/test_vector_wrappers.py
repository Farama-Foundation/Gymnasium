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
from gymnasium.vector import VectorEnv
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


@pytest.mark.parametrize(
    "autoreset_mode", [AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP]
)
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
    vectorization_mode: str = "sync",
    num_steps: int = 50,
):
    vector_wrapper = getattr(wrappers.vector, wrapper_name)
    wrapper_vector_env: VectorEnv = vector_wrapper(
        gym.make_vec(
            id=env_id, num_envs=num_envs, vectorization_mode=vectorization_mode
        ),
        **kwargs,
    )
    env_wrapper = getattr(wrappers, wrapper_name)
    vector_wrapper_env = gym.make_vec(
        id=env_id,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        wrappers=(lambda env: env_wrapper(env, **kwargs),),
    )

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

    for _ in range(num_steps):
        action = wrapper_vector_env.action_space.sample()
        wrapper_vector_step_returns = wrapper_vector_env.step(action)
        vector_wrapper_step_returns = vector_wrapper_env.step(action)

        for wrapper_vector_return, vector_wrapper_return in zip(
            wrapper_vector_step_returns, vector_wrapper_step_returns
        ):
            assert data_equivalence(wrapper_vector_return, vector_wrapper_return)

    wrapper_vector_env.close()
    vector_wrapper_env.close()
