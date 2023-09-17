"""Test suite for RecordVideo wrapper."""
import os
import shutil
from typing import List

import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def test_record_video_using_default_trigger():
    """Test RecordVideo using the default episode trigger."""
    env = gym.make("CartPole-v1", render_mode="rgb_array_list")
    env = RecordVideo(env, "videos")
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
    env = RecordVideo(env, "videos")
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


def test_record_video_frames_to_disk():
    env = gym.make("CartPole-v1", render_mode="rgb_array", disable_env_checker=True)
    env = gym.wrappers.RecordVideo(
        env, "videos", episode_trigger=lambda x: x % 100 == 0, frames_to_disk=True
    )
    ob_space = env.observation_space
    obs, info = env.reset()
    assert os.path.isdir(os.path.join("videos", "frames"))
    assert os.path.isfile(os.path.join("videos", "frames", "frame_0.png"))
    env.close()
    assert os.path.isdir("videos")
    assert any(file.endswith(".mp4") for file in os.listdir("videos"))
    assert not os.path.isdir(os.path.join("videos", "frames"))
    shutil.rmtree("videos")
    assert ob_space.contains(obs)
    assert isinstance(info, dict)


def test_record_video_step_trigger():
    """Test RecordVideo defining step trigger function."""
    env = gym.make("CartPole-v1", render_mode="rgb_array", disable_env_checker=True)
    env._max_episode_steps = 20
    env = RecordVideo(env, "videos", step_trigger=lambda x: x % 100 == 0)
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
    env = RecordVideo(
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
    env = RecordVideo(env, "videos", step_trigger=lambda x: x == 0, video_length=10)
    env.reset()
    for _ in range(10):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            break

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
    env = RecordVideo(env, "videos")
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
