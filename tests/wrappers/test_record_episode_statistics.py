"""Test suite for RecordEpisodeStatistics wrapper."""

import pytest

import gymnasium as gym
from gymnasium.wrappers import Autoreset, RecordEpisodeStatistics


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
@pytest.mark.parametrize("deque_size", [2, 5])
def test_record_episode_statistics(env_id, deque_size):
    env = gym.make(env_id, disable_env_checker=True)
    env = RecordEpisodeStatistics(env, deque_size)

    for _ in range(5):
        env.reset()
        assert env.episode_returns is not None and env.episode_lengths is not None
        assert env.episode_returns == 0.0
        assert env.episode_lengths == 0
        assert env.spec is not None
        for _ in range(env.spec.max_episode_steps):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                assert "episode" in info
                assert all([item in info["episode"] for item in ["r", "l", "t"]])
                break
    assert len(env.return_queue) == deque_size
    assert len(env.length_queue) == deque_size


def test_record_episode_statistics_with_autoreset():
    """Tests that the statistics restart for each episode of an autoreset environment."""
    env = gym.make("CartPole-v1", disable_env_checker=True)
    env = RecordEpisodeStatistics(Autoreset(env))
    env.reset(seed=0)
    env.action_space.seed(0)

    episode_lengths = []
    episode_length, autoreset = 0, False
    while len(episode_lengths) < 3:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        # The step after an episode ends resets the environment and isn't part of an episode
        episode_length = 0 if autoreset else episode_length + 1
        autoreset = terminated or truncated

        if autoreset:
            # CartPole's reward is 1 for every step
            assert info["episode"]["r"] == episode_length
            assert info["episode"]["l"] == episode_length
            episode_lengths.append(episode_length)
        else:
            assert "episode" not in info

    assert list(env.return_queue) == episode_lengths
    assert list(env.length_queue) == episode_lengths
