"""Test suite for TimeAwareobservationV0."""

from collections import OrderedDict

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.experimental.wrappers import TimeAwareObservationV0
from gymnasium.spaces import Box, Dict

NUM_STEPS = 20
SEED = 0


@pytest.mark.parametrize(
    "env",
    [
        gym.make("CartPole-v1", disable_env_checker=True),
        gym.make("CarRacing-v2", disable_env_checker=True),
    ],
)
def test_time_aware_observation_creation(env):
    """Test TimeAwareObservationV0 wrapper creation.

    This test checks if wrapped env with TimeAwareObservationV0
    is correctly created.
    """
    wrapped_env = TimeAwareObservationV0(env)
    obs, _ = wrapped_env.reset()

    assert isinstance(wrapped_env.observation_space, Dict)
    assert isinstance(obs, OrderedDict)
    assert np.all(obs["time"] == 0)
    assert env.observation_space == wrapped_env.observation_space["obs"]


@pytest.mark.parametrize("normalize_time", [True, False])
@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize(
    "env",
    [
        gym.make("CartPole-v1", disable_env_checker=True),
        gym.make("CarRacing-v2", disable_env_checker=True, continuous=False),
    ],
)
def test_time_aware_observation_step(env, flatten, normalize_time):
    """Test TimeAwareObservationV0 step.

    This test checks if wrapped env with TimeAwareObservationV0
    steps correctly.
    """
    env.action_space.seed(SEED)
    max_timesteps = env._max_episode_steps

    wrapped_env = TimeAwareObservationV0(
        env, flatten=flatten, normalize_time=normalize_time
    )
    wrapped_env.reset(seed=SEED)

    for timestep in range(1, NUM_STEPS):
        action = env.action_space.sample()
        observation, _, terminated, _, _ = wrapped_env.step(action)

        expected_time_obs = (
            timestep / max_timesteps if normalize_time else max_timesteps - timestep
        )

        if flatten:
            assert np.allclose(observation[-1], expected_time_obs)
        else:
            assert np.allclose(observation["time"], expected_time_obs)

        if terminated:
            break


@pytest.mark.parametrize(
    "env",
    [
        gym.make("CartPole-v1", disable_env_checker=True),
        gym.make("CarRacing-v2", disable_env_checker=True),
    ],
)
def test_time_aware_observation_creation_flatten(env):
    """Test TimeAwareObservationV0 wrapper creation with `flatten=True`.

    This test checks if wrapped env with TimeAwareObservationV0
    is correctly created when the `flatten` parameter is set to `True`.
    When flattened, the observation space should be a 1 dimension `Box`
    with time appended to the end.
    """
    wrapped_env = TimeAwareObservationV0(env, flatten=True)
    obs, _ = wrapped_env.reset()

    assert isinstance(wrapped_env.observation_space, Box)
    assert isinstance(obs, np.ndarray)
    assert env.observation_space == wrapped_env.time_aware_observation_space["obs"]
