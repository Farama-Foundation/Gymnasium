import os
import shutil
from unittest.mock import MagicMock

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.save_video import capped_cubic_video_schedule, save_video


def test_record_video_using_default_trigger():
    env = gym.make(
        "CartPole-v1", render_mode="rgb_array_list", disable_env_checker=True
    )

    env.reset()
    step_starting_index = 0
    episode_index = 0
    for step_index in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            save_video(
                env.render(),
                "videos",
                fps=env.metadata["render_fps"],
                step_starting_index=step_starting_index,
                episode_index=episode_index,
            )
            step_starting_index = step_index + 1
            episode_index += 1
            env.reset()

    env.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == sum(
        capped_cubic_video_schedule(i) for i in range(episode_index)
    )


def modulo_step_trigger(mod: int):
    def step_trigger(step_index):
        return step_index % mod == 0

    return step_trigger


def test_record_video_step_trigger():
    env = gym.make("CartPole-v1", render_mode="rgb_array_list")
    env._max_episode_steps = 20

    env.reset()
    step_starting_index = 0
    episode_index = 0
    for step_index in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            save_video(
                env.render(),
                "videos",
                fps=env.metadata["render_fps"],
                step_trigger=modulo_step_trigger(100),
                step_starting_index=step_starting_index,
                episode_index=episode_index,
            )
            step_starting_index = step_index + 1
            episode_index += 1
            env.reset()
    env.close()

    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == 2


def test_record_video_within_vector():
    step_trigger = modulo_step_trigger(100)
    n_steps = 199
    expected_video = 2

    envs = gym.make_vec(
        "CartPole-v1",
        num_envs=2,
        vectorization_mode="sync",
        render_mode="rgb_array_list",
    )
    envs.reset()
    episode_frames = []
    step_starting_index = 0
    episode_index = 0
    for step_index in range(n_steps):
        _, _, terminated, truncated, _ = envs.step(envs.action_space.sample())
        episode_frames.extend(envs.call("render")[0])

        if np.any(np.logical_or(terminated, truncated)):
            save_video(
                episode_frames,
                "videos",
                fps=envs.metadata["render_fps"],
                step_trigger=step_trigger,
                step_starting_index=step_starting_index,
                episode_index=episode_index,
            )
            episode_frames = []
            step_starting_index = step_index + 1
            episode_index += 1

            # TODO: fix this test (see https://github.com/openai/gym/issues/3054)
            if step_trigger(step_index):
                expected_video -= 1

    envs.close()

    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == expected_video


def test_save_video_snippet_durations(tmp_path, monkeypatch):
    clip_factory = MagicMock()
    monkeypatch.setattr("gymnasium.utils.save_video.ImageSequenceClip", clip_factory)
    frames = list(range(4))
    durations = [0.1, 0.2, 0.3, 0.4]
    save_video(
        frames,
        str(tmp_path),
        durations=durations,
        video_length=2,
        episode_trigger=lambda _: True,
        step_trigger=lambda step: step == 11,
        step_starting_index=10,
    )
    assert [call.args[0] for call in clip_factory.call_args_list] == [[0, 1], [2, 3]]
    assert [call.kwargs["durations"] for call in clip_factory.call_args_list] == [
        durations[:2],
        durations[2:],
    ]
    assert clip_factory.return_value.__exit__.call_count == 2


def test_save_video_invalid_inputs(tmp_path):
    with pytest.warns(UserWarning, match="No frames"):
        save_video([], str(tmp_path), fps=20)
    with pytest.raises(ValueError, match="one duration"):
        save_video([0, 1], str(tmp_path), durations=[1.0])
    with pytest.raises(ValueError, match="positive"):
        save_video([0], str(tmp_path), video_length=0, fps=20)
