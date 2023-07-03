import os
import re

import pytest

import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


class BrokenRecordableEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array_list"]}

    def __init__(self, render_mode="rgb_array_list"):
        self.render_mode = render_mode

    def render(self):
        pass


class UnrecordableEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

    def render(self):
        pass


def test_record_simple():
    env = gym.make(
        "CartPole-v1", render_mode="rgb_array_list", disable_env_checker=True
    )
    rec = VideoRecorder(env)
    env.reset()
    rec.capture_frame()

    rec.close()

    assert not rec.broken
    assert os.path.exists(rec.path)
    f = open(rec.path)
    assert os.fstat(f.fileno()).st_size > 100


def test_no_frames():
    env = BrokenRecordableEnv()
    rec = VideoRecorder(env)
    rec.close()
    assert rec.functional
    assert not os.path.exists(rec.path)


def test_record_unrecordable_method():
    error_message = (
        "Render mode is None, which is incompatible with RecordVideo."
        " Initialize your environment with a render_mode that returns an"
        " image, such as rgb_array."
    )
    with pytest.raises(ValueError, match=re.escape(error_message)):
        env = UnrecordableEnv()
        rec = VideoRecorder(env)
        assert not rec.enabled
        rec.close()


def test_record_breaking_render_method():
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Env returned None on `render()`. Disabling further rendering for video recorder by marking as disabled:"
        ),
    ):
        env = BrokenRecordableEnv()
        rec = VideoRecorder(env)
        rec.capture_frame()
        rec.close()
        assert rec.broken
        assert not os.path.exists(rec.path)


def test_text_envs():
    env = gym.make(
        "FrozenLake-v1", render_mode="rgb_array_list", disable_env_checker=True
    )
    video = VideoRecorder(env)
    try:
        env.reset()
        video.capture_frame()
        video.close()
    finally:
        os.remove(video.path)
