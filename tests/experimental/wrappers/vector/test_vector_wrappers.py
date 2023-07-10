"""Tests that the vectorised wrappers operate identically in `VectorEnv(Wrapper)` and `VectorWrapper(VectorEnv)`.

The exception is the data converter wrappers (`JaxToTorch`, `JaxToNumpy` and `NumpyToJax`)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.experimental import wrappers
from gymnasium.experimental.vector import VectorEnv
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import data_equivalence
from tests.testing_env import GenericTestEnv


@pytest.fixture
def custom_environments():
    gym.register(
        "CustomDictEnv-v0",
        lambda: GenericTestEnv(
            observation_space=Dict({"a": Box(0, 1), "b": Discrete(5)})
        ),
    )

    yield

    del gym.registry["CustomDictEnv-v0"]


@pytest.mark.parametrize("num_envs", (1, 3))
@pytest.mark.parametrize(
    "env_id, wrapper_name, kwargs",
    (
        ("CustomDictEnv-v0", "FilterObservationV0", {"filter_keys": ["a"]}),
        ("CartPole-v1", "FlattenObservationV0", {}),
        ("CarRacing-v2", "GrayscaleObservationV0", {}),
        # ("CarRacing-v2", "ResizeObservationV0", {"shape": (35, 45)}),
        ("CarRacing-v2", "ReshapeObservationV0", {"shape": (96, 48, 6)}),
        ("CartPole-v1", "RescaleObservationV0", {"min_obs": 0, "max_obs": 1}),
        ("CartPole-v1", "DtypeObservationV0", {"dtype": np.int32}),
        # ("CartPole-v1", "PixelObservationV0", {}),
        # ("CartPole-v1", "NormalizeObservationV0", {}),
        # ("CartPole-v1", "TimeAwareObservationV0", {}),
        # ("CartPole-v1", "FrameStackObservationV0", {}),
        # ("CartPole-v1", "DelayObservationV0", {}),
        ("MountainCarContinuous-v0", "ClipActionV0", {}),
        (
            "MountainCarContinuous-v0",
            "RescaleActionV0",
            {"min_action": 1, "max_action": 2},
        ),
        # ("CartPole-v1", "StickyActionV0", {}),
        ("CartPole-v1", "ClipRewardV0", {"min_reward": 0.25, "max_reward": 0.75}),
        # ("CartPole-v1", "NormalizeRewardV1", {}),
    ),
)
def test_vector_wrapper_equivalence(
    env_id: str,
    wrapper_name: str,
    kwargs: dict[str, Any],
    num_envs: int,
    custom_environments,
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


# ("CartPole-v1", "LambdaObservationV0", {"func": lambda obs: obs + 1}),
# ("CartPole-v1", "LambdaActionV0", {"func": lambda action: action + 1}),
# ("CartPole-v1", "LambdaRewardV0", {"func": lambda reward: reward + 1}),
# (vector.JaxToNumpyV0, {}, {}),
# (vector.JaxToTorchV0, {}, {}),
# (vector.NumpyToTorchV0, {}, {}),
# ("CartPole-v1", "RecordEpisodeStatisticsV0", {}),  # for the time taken in info, this is not equivalent for two instances
