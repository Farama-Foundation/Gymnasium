"""Test suite for AtariProcessing wrapper."""

import re

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing


pytest.importorskip("ale_py")


@pytest.mark.parametrize(
    "env, expected_obs_shape",
    [
        (gym.make("ALE/Pong-v5"), (210, 160, 3)),
        (
            AtariPreprocessing(
                gym.make("ALE/Pong-v5"),
                screen_size=84,
                grayscale_obs=True,
                frame_skip=1,
                noop_max=0,
            ),
            (84, 84),
        ),
        (
            AtariPreprocessing(
                gym.make("ALE/Pong-v5"),
                screen_size=84,
                grayscale_obs=False,
                frame_skip=1,
                noop_max=0,
            ),
            (84, 84, 3),
        ),
        (
            AtariPreprocessing(
                gym.make("ALE/Pong-v5"),
                screen_size=84,
                grayscale_obs=True,
                frame_skip=1,
                noop_max=0,
                grayscale_newaxis=True,
            ),
            (84, 84, 1),
        ),
        (
            AtariPreprocessing(
                gym.make("ALE/Pong-v5"),
                screen_size=(160, 210),
                grayscale_obs=False,
                frame_skip=1,
                noop_max=0,
                grayscale_newaxis=True,
            ),
            (210, 160, 3),
        ),
    ],
)
def test_atari_preprocessing_grayscale(env, expected_obs_shape):
    assert env.observation_space.shape == expected_obs_shape

    obs, _ = env.reset(seed=0)
    assert obs in env.observation_space

    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs in env.observation_space

    env.close()


@pytest.mark.parametrize("grayscale", [True, False])
@pytest.mark.parametrize("scaled", [True, False])
def test_atari_preprocessing_scale(grayscale, scaled, max_test_steps=10):
    # arbitrarily chosen number for stepping into env. and ensuring all observations are in the required range
    env = AtariPreprocessing(
        gym.make("ALE/Pong-v5"),
        screen_size=84,
        grayscale_obs=grayscale,
        scale_obs=scaled,
        frame_skip=1,
        noop_max=0,
    )

    obs, _ = env.reset()

    max_obs = 1 if scaled else 255
    assert np.all(0 <= obs) and np.all(obs <= max_obs)

    terminated, truncated, step_i = False, False, 0
    while not (terminated or truncated) and step_i <= max_test_steps:
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(0 <= obs) and np.all(obs <= max_obs)

        step_i += 1
    env.close()


def test_screen_size():
    env = gym.make("ALE/Pong-v5", frameskip=1)

    assert AtariPreprocessing(env).screen_size == (84, 84)
    assert AtariPreprocessing(env, screen_size=50).screen_size == (50, 50)
    assert AtariPreprocessing(env, screen_size=(100, 120)).screen_size == (100, 120)

    with pytest.raises(
        AssertionError, match="Expect the `screen_size` to be positive, actually: -1"
    ):
        AtariPreprocessing(env, screen_size=-1)

    with pytest.raises(
        AssertionError,
        match=re.escape("Expect the `screen_size` to be positive, actually: (-1, 10)"),
    ):
        AtariPreprocessing(env, screen_size=(-1, 10))

    env.close()
