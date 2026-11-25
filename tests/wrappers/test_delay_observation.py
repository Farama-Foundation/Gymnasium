"""Test suite for DelayObservation wrapper."""

import re
from copy import deepcopy

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.wrappers import DelayObservation
from gymnasium.wrappers.utils import create_zero_array
from tests.testing_env import GenericTestEnv
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


@pytest.mark.parametrize("delay", [0, 1, 3])
@pytest.mark.parametrize("reuse_buffer", [False, True])
@pytest.mark.parametrize("nested", [False, True], ids=["array", "nested"])
def test_delayed_observation_history(delay, reuse_buffer, nested):
    """Delaying observations preserves values when the environment reuses storage."""
    array_space = gym.spaces.Box(-100, 100, (2,), dtype=np.float32)
    buffer = np.empty(array_space.shape, dtype=array_space.dtype)
    observation_space = (
        gym.spaces.Dict(value=gym.spaces.Tuple((array_space,)))
        if nested
        else array_space
    )
    shared_observation = {"value": (buffer,)} if nested else buffer
    timestep = 0

    def observation():
        buffer[:] = [timestep, -timestep]
        return shared_observation if reuse_buffer else deepcopy(shared_observation)

    def reset(self, *, seed=None, options=None):
        nonlocal timestep
        timestep = options["start"]
        return observation(), {"reset": timestep}

    def step(self, action):
        nonlocal timestep
        timestep += 1
        return observation(), float(timestep), timestep == 7, False, {"step": timestep}

    env = DelayObservation(
        GenericTestEnv(
            observation_space=observation_space, reset_func=reset, step_func=step
        ),
        delay=delay,
    )
    zero = create_zero_array(observation_space)
    returned = []
    for start in [1, 20]:
        obs, info = env.reset(options={"start": start})
        assert info == {"reset": start}
        expected = [deepcopy(shared_observation)]
        assert data_equivalence(obs, zero if delay else expected[0])
        for index in range(1, 8):
            obs, reward, terminated, truncated, info = env.step(0)
            expected.append(deepcopy(shared_observation))
            assert data_equivalence(
                obs, zero if index < delay else expected[index - delay]
            )
            assert obs in observation_space
            assert (reward, terminated, truncated, info) == (
                float(start + index),
                start + index == 7,
                False,
                {"step": start + index},
            )
            if delay:
                returned.append((obs, deepcopy(obs)))
            elif reuse_buffer:
                assert obs is shared_observation

    for obs, snapshot in returned:
        assert data_equivalence(obs, snapshot)
    env.close()
