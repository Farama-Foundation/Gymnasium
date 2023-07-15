import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, VectorListInfo


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
@pytest.mark.parametrize("deque_size", [2, 5])
def test_record_episode_statistics(env_id, deque_size):
    env = gym.make(env_id, disable_env_checker=True)
    env = RecordEpisodeStatistics(env, deque_size)

    for n in range(5):
        env.reset()
        assert env.episode_returns is not None and env.episode_lengths is not None
        assert env.episode_returns[0] == 0.0
        assert env.episode_lengths[0] == 0
        assert env.spec is not None
        for t in range(env.spec.max_episode_steps):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                assert "episode" in info
                assert all([item in info["episode"] for item in ["r", "l", "t"]])
                break
    assert len(env.return_queue) == deque_size
    assert len(env.length_queue) == deque_size


def test_record_episode_statistics_reset_info():
    env = gym.make("CartPole-v1", disable_env_checker=True)
    env = RecordEpisodeStatistics(env)
    ob_space = env.observation_space
    obs, info = env.reset()
    assert ob_space.contains(obs)
    assert isinstance(info, dict)


@pytest.mark.parametrize(
    ("num_envs", "vectorization_mode"),
    [(1, "sync"), (1, "async"), (4, "sync"), (4, "async")],
)
def test_record_episode_statistics_with_vectorenv(num_envs, vectorization_mode):
    envs = gym.make_vec(
        "CartPole-v1",
        render_mode=None,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        disable_env_checker=True,
    )
    envs = RecordEpisodeStatistics(envs)
    if vectorization_mode == "async":
        max_episode_step = envs.unwrapped.env_fns[0]().spec.max_episode_steps
    else:
        max_episode_step = envs.unwrapped.envs[0].spec.max_episode_steps

    envs.reset()
    for _ in range(max_episode_step + 1):
        _, _, terminateds, truncateds, infos = envs.step(envs.action_space.sample())
        if any(terminateds) or any(truncateds):
            assert "episode" in infos
            assert "_episode" in infos
            assert all(infos["_episode"] == np.bitwise_or(terminateds, truncateds))
            assert all([item in infos["episode"] for item in ["r", "l", "t"]])
            break
        else:
            assert "episode" not in infos
            assert "_episode" not in infos


@pytest.mark.skip(reason="With new vector environment, not possible to incorrectly do")
def test_wrong_wrapping_order():
    envs = gym.make_vec("CartPole-v1", num_envs=3)
    wrapped_env = RecordEpisodeStatistics(VectorListInfo(envs))
    wrapped_env.reset()

    with pytest.raises(AssertionError):
        wrapped_env.step(wrapped_env.action_space.sample())
