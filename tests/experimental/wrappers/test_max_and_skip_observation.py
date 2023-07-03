"""Test suite for MaxAndSkipObservationV0."""
import re

import pytest

import gymnasium as gym
from gymnasium.experimental.wrappers import MaxAndSkipObservationV0


def test_max_and_skip_obs(skip: int = 4):
    """Test MaxAndSkipObservationV0."""
    env = gym.make("CartPole-v1")

    env = MaxAndSkipObservationV0(env, skip=skip)

    obs, _ = env.reset()
    assert obs in env.observation_space

    for i in range(10):
        obs, _, _, _, _ = env.step(env.action_space.sample())
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
        MaxAndSkipObservationV0(env, skip=1.0)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The skip value needs to be equal or greater than two, actual value: 0"
        ),
    ):
        MaxAndSkipObservationV0(env, skip=0)
