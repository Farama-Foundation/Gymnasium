"""Test suite for RecordVideo wrapper."""

import os
import shutil

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import HumanRendering, RecordVideo, RenderCollection
from tests.wrappers.test_human_rendering import CountingRGBEnv


@pytest.mark.parametrize(
    "human_inside,reuse_frames", [(True, True), (True, False), (False, False)]
)
def test_record_video_with_rgb_human_rendering(tmp_path, human_inside, reuse_frames):
    """Encode and read back displayed frames in both supported wrapper orders."""
    from moviepy import VideoFileClip

    base = CountingRGBEnv()
    # The outer cache protects reused buffers; an inner recorder sees base frames directly.
    base.copy_frames = not reuse_frames
    base.metadata = {**base.metadata, "render_fps": 20}
    folder = str(tmp_path / "recording")
    if human_inside:
        env = RecordVideo(
            HumanRendering(base, return_rgb_array=True),
            folder,
            episode_trigger=lambda _: True,
        )
    else:
        env = HumanRendering(
            RecordVideo(base, folder, episode_trigger=lambda _: True),
            return_rgb_array=True,
        )
    try:
        env.reset()
        frames = [env.render()]
        for _ in range(3):
            env.step(0)
            frames.append(env.render())
        if human_inside:
            assert base.render_calls == 4
    finally:
        env.close()
    videos = list((tmp_path / "recording").glob("*.mp4"))
    assert len(videos) == 1
    with VideoFileClip(str(videos[0])) as clip:
        decoded = list(clip.iter_frames())
    assert len(decoded) == 4
    for expected, actual in zip(frames, decoded, strict=True):
        assert actual.shape == expected.shape == (16, 24, 3)
        # H.264 is lossy; the temporal colors and spatial orientation must survive.
        np.testing.assert_allclose(actual.astype(float), expected.astype(float), atol=8)


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


def test_incompatible_render_mode():
    """Construction with a render mode that yields no image fails cleanly.

    The message used to be split across two arguments of `ValueError`, so
    `str(e)` printed a tuple and the sentence telling the user what to do
    arrived quoted inside it. `__del__` then ran on the half-built wrapper and
    raised an AttributeError of its own, which Python reports as an ignored
    exception on top of the real one.
    """
    env = gym.make("CartPole-v1")  # render_mode is None

    with pytest.raises(ValueError) as exc_info:
        RecordVideo(env, "videos")

    assert len(exc_info.value.args) == 1
    assert "such as rgb_array" in str(exc_info.value)

    env.close()

    # `__del__` on a wrapper whose `__init__` never reached `recorded_frames`.
    RecordVideo.__new__(RecordVideo).__del__()
