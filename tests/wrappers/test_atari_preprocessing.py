"""Test suite for AtariProcessing wrapper."""

import numpy as np
import pytest

from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import AtariPreprocessing
from tests.testing_env import GenericTestEnv


class AleTesting:
    """A testing implementation for the ALE object in atari games."""

    grayscale_obs_space = Box(low=0, high=255, shape=(210, 160), dtype=np.uint8, seed=1)
    rgb_obs_space = Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8, seed=1)

    def lives(self) -> int:
        """Returns the number of lives in the atari game."""
        return 1

    def getScreenGrayscale(self, buffer: np.ndarray):
        """Updates the buffer with a random grayscale observation."""
        buffer[...] = self.grayscale_obs_space.sample()

    def getScreenRGB(self, buffer: np.ndarray):
        """Updates the buffer with a random rgb observation."""
        buffer[...] = self.rgb_obs_space.sample()


class AtariTestingEnv(GenericTestEnv):
    """A testing environment to replicate the atari (ale-py) environments."""

    def __init__(self):
        super().__init__(
            observation_space=Box(
                low=0, high=255, shape=(210, 160, 3), dtype=np.uint8, seed=1
            ),
            action_space=Discrete(3, seed=1),
        )
        self.ale = AleTesting()

    def get_action_meanings(self):
        """Returns the meanings of each of the actions available to the agent. First index must be 'NOOP'."""
        return ["NOOP", "UP", "DOWN"]


@pytest.mark.parametrize(
    "env, expected_obs_shape",
    [
        (AtariTestingEnv(), (210, 160, 3)),
        (
            AtariPreprocessing(
                AtariTestingEnv(),
                screen_size=84,
                grayscale_obs=True,
                frame_skip=1,
                noop_max=0,
            ),
            (84, 84),
        ),
        (
            AtariPreprocessing(
                AtariTestingEnv(),
                screen_size=84,
                grayscale_obs=False,
                frame_skip=1,
                noop_max=0,
            ),
            (84, 84, 3),
        ),
        (
            AtariPreprocessing(
                AtariTestingEnv(),
                screen_size=84,
                grayscale_obs=True,
                frame_skip=1,
                noop_max=0,
                grayscale_newaxis=True,
            ),
            (84, 84, 1),
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
        AtariTestingEnv(),
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
