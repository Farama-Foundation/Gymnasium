"""Test suite for RecordVideo wrapper."""

import os
import shutil

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers.vector import RecordVideo


def test_video_folder_and_filenames(
    video_folder="custom_video_folder", name_prefix="video-prefix"
):
    def make_env():
        return gym.make("CartPole-v1", render_mode="rgb_array")

    n_envs = 10
    env = gym.vector.SyncVectorEnv([make_env for _ in range(n_envs)])
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda x: x in [1, 4],
        step_trigger=lambda x: x in [0, 25],
        video_aspect_ratio=(1, 1),
    )

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(100):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if np.any(terminated) or np.any(truncated):
            env.reset()
    env.close()

    assert os.path.isdir(video_folder)
    mp4_files = {file for file in os.listdir(video_folder) if file.endswith(".mp4")}
    shutil.rmtree(video_folder)
    assert mp4_files == {
        "video-prefix-step-0.mp4",  # step triggers
        "video-prefix-step-25.mp4",
        "video-prefix-episode-1.mp4",  # episode triggers
        "video-prefix-episode-4.mp4",
    }


@pytest.mark.parametrize("episodic_trigger", [None, lambda x: x in [0, 3, 5, 10, 12]])
def test_episodic_trigger(episodic_trigger):
    """Test RecordVideo using the default episode trigger."""

    def make_env():
        return gym.make("CartPole-v1", render_mode="rgb_array")

    n_envs = 10
    env = gym.vector.SyncVectorEnv([make_env for _ in range(n_envs)])
    env = RecordVideo(
        env,
        "videos",
        episode_trigger=episodic_trigger,
        video_aspect_ratio=(1, 1),
    )
    env.reset()
    episode_count = 0
    for _ in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if np.any(terminated) or np.any(truncated):
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


def test_step_trigger():
    """Test RecordVideo defining step trigger function."""
    n_envs = 10

    def make_env():
        return gym.make("CartPole-v1", render_mode="rgb_array")

    env = gym.vector.SyncVectorEnv([make_env for _ in range(n_envs)])
    env = RecordVideo(
        env,
        "videos",
        step_trigger=lambda x: x % 100 == 0,
        video_aspect_ratio=(1, 1),
    )
    env.reset()
    for _ in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if np.any(terminated) or np.any(truncated):
            env.reset()
    env.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == 2


def test_both_episodic_and_step_trigger():
    """Test RecordVideo defining both step and episode trigger functions."""
    n_envs = 10

    def make_env():
        return gym.make("CartPole-v1", render_mode="rgb_array")

    env = gym.vector.SyncVectorEnv([make_env for _ in range(n_envs)])
    env = RecordVideo(
        env,
        "videos",
        step_trigger=lambda x: x == 100,
        episode_trigger=lambda x: x == 0 or x == 3,
        video_aspect_ratio=(1, 1),
    )

    env.reset(seed=123)
    env.action_space.seed(123)
    for i in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if np.any(terminated) or np.any(truncated):
            env.reset()
    env.close()

    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == 3


def test_video_length(video_length: int = 10):
    """Test if argument video_length of RecordVideo works properly."""
    n_envs = 10

    def make_env():
        return gym.make("CartPole-v1", render_mode="rgb_array")

    env = gym.vector.SyncVectorEnv([make_env for _ in range(n_envs)])
    env = RecordVideo(
        env,
        "videos",
        step_trigger=lambda x: x == 0,
        video_length=video_length,
        video_aspect_ratio=(1, 1),
    )

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(video_length):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if np.any(terminated) or np.any(truncated):
            break

    # check that the environment is still recording then take a step to take the number of steps > video length
    assert env.recording
    env.step(env.action_space.sample())
    assert not env.recording
    env.close()

    # check that only one video is recorded
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    assert len(mp4_files) == 1
    shutil.rmtree("videos")
