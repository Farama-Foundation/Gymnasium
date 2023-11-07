"""Test suite for RecordEpisodeStatistics wrapper."""

import pytest

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
@pytest.mark.parametrize("deque_size", [2, 5])
def test_record_episode_statistics(env_id, deque_size):
    env = gym.make(env_id, disable_env_checker=True)
    env = RecordEpisodeStatistics(env, deque_size)

    for n in range(5):
        env.reset()
        assert env.episode_returns is not None and env.episode_lengths is not None
        assert env.episode_returns == 0.0
        assert env.episode_lengths == 0
        assert env.spec is not None
        for t in range(env.spec.max_episode_steps):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                assert "episode" in info
                assert all([item in info["episode"] for item in ["r", "l", "t"]])
                break
    assert len(env.return_queue) == deque_size
    assert len(env.length_queue) == deque_size
