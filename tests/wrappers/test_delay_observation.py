"""Test suite for DelayObservation wrapper."""

import re

import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.wrappers import DelayObservation
from gymnasium.wrappers.utils import create_zero_array
from tests.wrappers.utils import SEED, TESTING_OBS_ENVS, TESTING_OBS_ENVS_IDS


@pytest.mark.parametrize("env", TESTING_OBS_ENVS, ids=TESTING_OBS_ENVS_IDS)
def test_env_obs(env, delay: int = 3, extra_steps: int = 4):
    """Tests the delay observation wrapper."""
    env.action_space.seed(SEED)
    obs, _ = env.reset(seed=SEED)

    undelayed_obs = [obs]
    for _ in range(delay + extra_steps):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        undelayed_obs.append(obs)

    env = DelayObservation(env, delay=delay)
    example_zero_obs = create_zero_array(env.observation_space)
    env.action_space.seed(SEED)
    obs, _ = env.reset(seed=SEED)
    assert data_equivalence(obs, example_zero_obs)
    assert obs in env.observation_space

    delayed_obs = [obs]
    for i in range(delay + extra_steps):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        delayed_obs.append(obs)
        assert obs in env.observation_space

        if i < delay - 1:
            assert data_equivalence(obs, example_zero_obs)

    assert data_equivalence(delayed_obs[delay:], undelayed_obs[:-delay])


@pytest.mark.parametrize("delay", [1, 2, 3, 4])
def test_delay_values(delay):
    """Test the possible delay values for the DelayObservation wrapper."""
    env = gym.make("CartPole-v1")
    first_obs, _ = env.reset(seed=123)

    env = DelayObservation(gym.make("CartPole-v1"), delay=delay)
    zero_obs = create_zero_array(env.observation_space)
    obs, _ = env.reset(seed=123)
    assert data_equivalence(obs, zero_obs)
    for _ in range(delay - 1):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert data_equivalence(obs, zero_obs)

    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert data_equivalence(first_obs, obs)


def test_delay_failures():
    """Test errors raised by DelayObservation wrapper."""
    env = gym.make("CartPole-v1")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The delay is expected to be an integer, actual type: <class 'float'>"
        ),
    ):
        DelayObservation(env, delay=1.0)

    with pytest.raises(
        ValueError,
        match=re.escape("The delay needs to be greater than zero, actual value: -1"),
    ):
        DelayObservation(env, delay=-1)
