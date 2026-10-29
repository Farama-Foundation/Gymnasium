"""Test suite for MaxAndSkipObservation wrapper."""

import re

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation


def test_max_and_skip_obs(skip: int = 4):
    """Test MaxAndSkipObservationV0."""
    env = gym.make("CartPole-v1")

    env = MaxAndSkipObservation(env, skip=skip)

    obs, _ = env.reset()
    assert obs in env.observation_space

    for _ in range(10):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert obs in env.observation_space

        if term or trunc:
            obs, _ = env.reset()
            assert obs in env.observation_space


def test_skip_size_failures():
    """Test the error raised by the MaxAndSkipObservation."""
    env = gym.make("CartPole-v1")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The skip is expected to be an integer, actual type: <class 'float'>"
        ),
    ):
        MaxAndSkipObservation(env, skip=1.0)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The skip value needs to be equal or greater than two, actual value: 0"
        ),
    ):
        MaxAndSkipObservation(env, skip=0)


class _EarlyEndingEnv(gym.Env):
    """Return known observations using a single mutable array until the chosen step."""

    def __init__(self, end_step, truncate):
        """Set the episode length and which ending flag to return."""
        self.observation_space = gym.spaces.Box(-100, 100, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.frames = np.array(
            [[-9, -1], [-2, -8], [-7, -3], [-4, -6], [-8, -5], [-6, -9]],
            dtype=np.float32,
        )
        self.end_step = end_step
        self.truncate = truncate
        self.observation_buffer = np.empty(2, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Start a new episode with an observation distinct from every step."""
        super().reset(seed=seed)
        self.steps = 0
        self.observation_buffer[:] = 100
        return self.observation_buffer, {"reset": True}

    def step(self, action):
        """Overwrite the observation buffer and stop exactly at the chosen step."""
        assert self.steps < self.end_step
        assert action == 1
        self.observation_buffer[:] = self.frames[self.steps]
        self.steps += 1
        ended = self.steps == self.end_step
        return (
            self.observation_buffer,
            float(self.steps),
            ended and not self.truncate,
            ended and self.truncate,
            {"step": self.steps},
        )


@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize(
    "skip,end_step",
    [(skip, end_step) for skip in (2, 3, 4) for end_step in range(1, skip + 1)],
)
def test_max_and_skip_early_ending(skip, end_step, truncate):
    """Pool the last available frames, including when only one step is taken."""
    base_env = _EarlyEndingEnv(end_step, truncate)
    env = MaxAndSkipObservation(base_env, skip=skip)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(1)

    expected = base_env.frames[max(0, end_step - 2) : end_step].max(axis=0)
    np.testing.assert_array_equal(obs, expected)
    assert obs.dtype == np.float32
    assert reward == sum(range(1, end_step + 1))
    assert terminated is (not truncate)
    assert truncated is truncate
    assert info == {"step": end_step}
    assert base_env.steps == end_step


@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize("reset_before_ending", [False, True])
def test_max_and_skip_discards_previous_frames(truncate, reset_before_ending):
    """An early ending must not pool frames from a previous call or episode."""
    base_env = _EarlyEndingEnv(end_step=6, truncate=truncate)
    env = MaxAndSkipObservation(base_env, skip=4)
    env.reset()
    previous_obs, reward, terminated, truncated, info = env.step(1)
    np.testing.assert_array_equal(previous_obs, base_env.frames[2:4].max(axis=0))
    assert reward == 10.0
    assert not terminated and not truncated
    assert info == {"step": 4}

    if reset_before_ending:
        base_env.end_step = 1
        env.reset()
        expected = base_env.frames[0]
        expected_reward = 1.0
    else:
        expected = base_env.frames[4:6].max(axis=0)
        expected_reward = 11.0

    obs, reward, terminated, truncated, info = env.step(1)

    np.testing.assert_array_equal(obs, expected)
    np.testing.assert_array_equal(previous_obs, base_env.frames[2:4].max(axis=0))
    assert reward == expected_reward
    assert terminated is (not truncate)
    assert truncated is truncate
    assert info == {"step": base_env.end_step}
