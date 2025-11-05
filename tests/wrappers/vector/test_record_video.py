"""Test suite for RecordVideo wrapper."""

import os
import shutil

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers.vector import RecordVideo


@pytest.mark.parametrize("autoreset_mode", list(gym.vector.AutoresetMode.__iter__()))
def test_video_folder_and_filenames(
    autoreset_mode: gym.vector.AutoresetMode,
    video_folder: str = "custom_video_folder",
    name_prefix: str = "video-prefix",
    n_envs: int = 10,
):
    envs = gym.make_vec(
        "CartPole-v1",
        num_envs=n_envs,
        render_mode="rgb_array",
        vectorization_mode=gym.VectorizeMode.SYNC,
        vector_kwargs={"autoreset_mode": autoreset_mode},
    )
    envs = RecordVideo(
        envs,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda x: x in [1, 4],
        step_trigger=lambda x: x in [0, 30],
        video_aspect_ratio=(1, 1),
    )

    envs.reset(seed=123)
    envs.action_space.seed(123)
    for _ in range(150):
        actions = envs.action_space.sample()
        _, _, terminations, truncations, _ = envs.step(actions)

        autoresets = np.logical_or(terminations, truncations)
        if autoreset_mode == AutoresetMode.DISABLED and np.any(autoresets):
            envs.reset(options={"reset_mask": autoresets})

    envs.close()

    assert os.path.isdir(video_folder)
    mp4_files = {file for file in os.listdir(video_folder) if file.endswith(".mp4")}
    shutil.rmtree(video_folder)
    assert mp4_files == {
        "video-prefix-step-0.mp4",  # step triggers
        "video-prefix-step-30.mp4",
        "video-prefix-episode-1.mp4",  # episode triggers
        "video-prefix-episode-4.mp4",
    }


@pytest.mark.parametrize("autoreset_mode", list(gym.vector.AutoresetMode.__iter__()))
@pytest.mark.parametrize("episodic_trigger", [None, lambda x: x in [0, 3, 5, 10, 12]])
def test_episodic_trigger(autoreset_mode, episodic_trigger, n_envs=10):
    """Test RecordVideo using the default episode trigger."""
    envs = gym.make_vec(
        "CartPole-v1",
        num_envs=n_envs,
        render_mode="rgb_array",
        vectorization_mode=gym.VectorizeMode.SYNC,
        vector_kwargs={"autoreset_mode": autoreset_mode},
    )
    envs = RecordVideo(
        envs,
        "videos",
        episode_trigger=episodic_trigger,
        video_aspect_ratio=(1, 1),
    )
    envs.reset(seed=123)
    envs.action_space.seed(123)
    episode_count = 0
    for _ in range(199):
        action = envs.action_space.sample()
        _, _, terminations, truncations, _ = envs.step(action)
        if terminations[0] or truncations[0]:
            episode_count += 1

        autoresets = np.logical_or(terminations, truncations)
        if autoreset_mode == AutoresetMode.DISABLED and np.any(autoresets):
            envs.reset(options={"reset_mask": autoresets})
    envs.close()

    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert envs.episode_trigger is not None
    assert len(mp4_files) == sum(
        envs.episode_trigger(i) for i in range(episode_count + 1)
    )


@pytest.mark.parametrize("autoreset_mode", list(gym.vector.AutoresetMode.__iter__()))
def test_step_trigger(autoreset_mode, n_envs=10):
    """Test RecordVideo defining step trigger function."""
    envs = gym.make_vec(
        "CartPole-v1",
        num_envs=n_envs,
        render_mode="rgb_array",
        vectorization_mode=gym.VectorizeMode.SYNC,
        vector_kwargs={"autoreset_mode": autoreset_mode},
    )
    envs = RecordVideo(
        envs,
        "videos",
        step_trigger=lambda x: x % 100 == 0,
        video_aspect_ratio=(1, 1),
    )
    envs.reset(seed=123)
    envs.action_space.seed(123)
    for _ in range(199):
        action = envs.action_space.sample()
        _, _, terminations, truncations, _ = envs.step(action)

        autoresets = np.logical_or(terminations, truncations)
        if autoreset_mode == AutoresetMode.DISABLED and np.any(autoresets):
            envs.reset(options={"reset_mask": autoresets})

    envs.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == 2


@pytest.mark.parametrize("autoreset_mode", list(gym.vector.AutoresetMode.__iter__()))
def test_both_episodic_and_step_trigger(autoreset_mode, n_envs=10):
    """Test RecordVideo defining both step and episode trigger functions."""
    envs = gym.make_vec(
        "CartPole-v1",
        num_envs=n_envs,
        render_mode="rgb_array",
        vectorization_mode=gym.VectorizeMode.SYNC,
        vector_kwargs={"autoreset_mode": autoreset_mode},
    )
    envs = RecordVideo(
        envs,
        "videos",
        step_trigger=lambda x: x == 100,
        episode_trigger=lambda x: x == 0 or x == 3,
        video_aspect_ratio=(1, 1),
    )

    envs.reset(seed=123)
    envs.action_space.seed(123)
    for i in range(199):
        action = envs.action_space.sample()
        _, _, terminations, truncations, _ = envs.step(action)

        autoresets = np.logical_or(terminations, truncations)
        if autoreset_mode == AutoresetMode.DISABLED and np.any(autoresets):
            envs.reset(options={"reset_mask": autoresets})
    envs.close()

    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == 3


@pytest.mark.parametrize("autoreset_mode", list(gym.vector.AutoresetMode.__iter__()))
def test_video_length(autoreset_mode, video_length: int = 10, n_envs=10):
    """Test if argument video_length of RecordVideo works properly."""
    envs = gym.make_vec(
        "CartPole-v1",
        num_envs=n_envs,
        render_mode="rgb_array",
        vectorization_mode=gym.VectorizeMode.SYNC,
        vector_kwargs={"autoreset_mode": autoreset_mode},
    )
    envs = RecordVideo(
        envs,
        "videos",
        step_trigger=lambda x: x == 0,
        video_length=video_length,
        video_aspect_ratio=(1, 1),
    )

    envs.reset(seed=123)
    envs.action_space.seed(123)
    for _ in range(video_length):
        _, _, terminations, truncations, _ = envs.step(envs.action_space.sample())

        if terminations[0] or truncations[0]:
            break

        autoresets = np.logical_or(terminations, truncations)
        if autoreset_mode == AutoresetMode.DISABLED and np.any(autoresets):
            envs.reset(options={"reset_mask": autoresets})

    # check that the environment is still recording then take a step to take the number of steps > video length
    assert envs.recording
    envs.step(envs.action_space.sample())
    assert not envs.recording
    envs.close()

    # check that only one video is recorded
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == 1
