"""Test suite for RecordVideo wrapper."""

import os
import shutil

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RenderCollection


def test_video_folder_and_filenames(
    video_folder="custom_video_folder", name_prefix="video-prefix"
):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda x: x in [1, 4],
        step_trigger=lambda x: x in [0, 25],
    )

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(100):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
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
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, "videos", episode_trigger=episodic_trigger)

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


def test_step_trigger():
    """Test RecordVideo defining step trigger function."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
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


def test_both_episodic_and_step_trigger():
    """Test RecordVideo defining both step and episode trigger functions."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(
        env,
        "videos",
        step_trigger=lambda x: x == 100,
        episode_trigger=lambda x: x == 0 or x == 3,
    )
    # episode reset time steps: 0, 18, 44, 55, 80, 103, 117, 143, 173, 191
    # steps recorded: 0-18, 55-80, 100-103

    env.reset(seed=123)
    env.action_space.seed(123)
    for i in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    env.close()

    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == 3


def test_video_length(video_length: int = 10):
    """Test if argument video_length of RecordVideo works properly."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(
        env, "videos", step_trigger=lambda x: x == 0, video_length=video_length
    )

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(video_length):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
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


def test_with_rgb_array_list(n_steps: int = 10):
    """Test if `env.render` works with RenderCollection and RecordVideo."""
    # fyi, can't work as a `pytest.mark.parameterize`
    env = RecordVideo(
        RenderCollection(gym.make("CartPole-v1", render_mode="rgb_array")), "videos"
    )
    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(n_steps):
        env.step(env.action_space.sample())

    render_out = env.render()
    assert isinstance(render_out, list)
    assert len(render_out) == n_steps + 1
    assert all(isinstance(render, np.ndarray) for render in render_out)
    assert all(render.ndim == 3 for render in render_out)

    render_out = env.render()
    assert isinstance(render_out, list)
    assert len(render_out) == 0

    env.close()
    shutil.rmtree("videos")

    # Test in reverse order
    env = RenderCollection(
        RecordVideo(gym.make("CartPole-v1", render_mode="rgb_array"), "videos")
    )
    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(n_steps):
        env.step(env.action_space.sample())

    render_out = env.render()
    assert isinstance(render_out, list)
    assert len(render_out) == n_steps + 1
    assert all(isinstance(render, np.ndarray) for render in render_out)
    assert all(render.ndim == 3 for render in render_out)

    render_out = env.render()
    assert isinstance(render_out, list)
    assert len(render_out) == 0

    env.close()
    shutil.rmtree("videos")
