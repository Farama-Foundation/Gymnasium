"""Test suite for DictInfoTolist wrapper."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers.vector import DictInfoToList, RecordEpisodeStatistics


ENV_ID = "CartPole-v1"
NUM_ENVS = 3
ENV_STEPS = 50
SEED = 42


def test_usage_in_vector_env():
    env = gym.make(ENV_ID, disable_env_checker=True)
    vector_env = gym.make_vec(ENV_ID, num_envs=NUM_ENVS, vectorization_mode="sync")

    DictInfoToList(vector_env)

    with pytest.raises(AssertionError):
        DictInfoToList(env)


def test_info_to_list():
    env_to_wrap = gym.make_vec(ENV_ID, num_envs=NUM_ENVS, vectorization_mode="sync")
    wrapped_env = DictInfoToList(env_to_wrap)
    wrapped_env.action_space.seed(SEED)
    _, info = wrapped_env.reset(seed=SEED)
    assert isinstance(info, list)
    assert len(info) == NUM_ENVS

    for _ in range(ENV_STEPS):
        action = wrapped_env.action_space.sample()
        _, _, terminateds, truncateds, list_info = wrapped_env.step(action)
        for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
            if terminated or truncated:
                assert "final_observation" in list_info[i]
            else:
                assert "final_observation" not in list_info[i]


def test_info_to_list_statistics():
    env_to_wrap = gym.make_vec(ENV_ID, num_envs=NUM_ENVS, vectorization_mode="sync")
    wrapped_env = DictInfoToList(RecordEpisodeStatistics(env_to_wrap))
    _, info = wrapped_env.reset(seed=SEED)
    wrapped_env.action_space.seed(SEED)
    assert isinstance(info, list)
    assert len(info) == NUM_ENVS

    for _ in range(ENV_STEPS):
        action = wrapped_env.action_space.sample()
        _, _, terminateds, truncateds, list_info = wrapped_env.step(action)
        for i, (terminated, truncated) in enumerate(zip(terminateds, truncateds)):
            if terminated or truncated:
                assert "episode" in list_info[i]
                for stats in ["r", "l", "t"]:
                    assert stats in list_info[i]["episode"]
                    assert np.isscalar(list_info[i]["episode"][stats])
            else:
                assert "episode" not in list_info[i]
