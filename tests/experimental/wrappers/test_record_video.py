"""Test suite for RecordVideoV0."""
import os
import shutil
from typing import List

import gymnasium as gym
from gymnasium.experimental.wrappers import RecordVideoV0


def test_record_video_using_default_trigger():
    """Test RecordVideo using the default episode trigger."""
    env = gym.make("CartPole-v1", render_mode="rgb_array_list")
    env = RecordVideoV0(env, "videos")
    env.reset()
    episode_count = 0
    for _ in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
            episode_count += 1

    env.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    assert env.episode_trigger is not None
    assert len(mp4_files) == sum(
        env.episode_trigger(i) for i in range(episode_count + 1)
    )
    shutil.rmtree("videos")


def test_record_video_while_rendering():
    """Test RecordVideo while calling render and using a _list render mode."""
    env = gym.make("FrozenLake-v1", render_mode="rgb_array_list")
    env = RecordVideoV0(env, "videos")
    env.reset()
    episode_count = 0
    for _ in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
            episode_count += 1

    env.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    assert env.episode_trigger is not None
    assert len(mp4_files) == sum(
        env.episode_trigger(i) for i in range(episode_count + 1)
    )
    shutil.rmtree("videos")


def test_record_video_step_trigger():
    """Test RecordVideo defining step trigger function."""
    env = gym.make("CartPole-v1", render_mode="rgb_array", disable_env_checker=True)
    env._max_episode_steps = 20
    env = RecordVideoV0(env, "videos", step_trigger=lambda x: x % 100 == 0)
    env.reset()
    for _ in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    env.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == 2


def test_record_video_both_trigger():
    """Test RecordVideo defining both step and episode trigger functions."""
    env = gym.make(
        "CartPole-v1", render_mode="rgb_array_list", disable_env_checker=True
    )
    env._max_episode_steps = 20
    env = RecordVideoV0(
        env,
        "videos",
        step_trigger=lambda x: x == 100,
        episode_trigger=lambda x: x == 0 or x == 3,
    )
    env.reset()
    for _ in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    env.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == 3


def test_record_video_length():
    """Test if argument video_length of RecordVideo works properly."""
    env = gym.make("CartPole-v1", render_mode="rgb_array_list")
    env._max_episode_steps = 20
    env = RecordVideoV0(env, "videos", step_trigger=lambda x: x == 0, video_length=10)
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)

    assert env.recording
    action = env.action_space.sample()
    env.step(action)
    assert not env.recording
    env.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    assert len(mp4_files) == 1
    shutil.rmtree("videos")


def test_rendering_works():
    """Test if render output is as expected when the env is wrapped with RecordVideo."""
    env = gym.make("CartPole-v1", render_mode="rgb_array_list")
    env._max_episode_steps = 20
    env = RecordVideoV0(env, "videos")
    env.reset()
    n_steps = 10
    for _ in range(n_steps):
        action = env.action_space.sample()
        env.step(action)

    render_out = env.render()
    assert isinstance(render_out, List)
    assert len(render_out) == n_steps + 1
    render_out = env.render()
    assert isinstance(render_out, List)
    assert len(render_out) == 0
    env.close()
    shutil.rmtree("videos")


def make_env(gym_id, idx, **kwargs):
    """Utility function to make an env and wrap it with RecordVideo only the first time."""

    def thunk():
        env = gym.make(gym_id, disable_env_checker=True, **kwargs)
        env._max_episode_steps = 20
        if idx == 0:
            env = RecordVideoV0(env, "videos", step_trigger=lambda x: x % 100 == 0)
        return env

    return thunk


def test_record_video_within_vector():
    """Test RecordVideo used as env of SyncVectorEnv."""
    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", i, render_mode="rgb_array") for i in range(2)]
    )
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs.reset()
    for i in range(199):
        _, _, _, _, infos = envs.step(envs.action_space.sample())

    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    assert len(mp4_files) == 2
    shutil.rmtree("videos")
