"""Test suite for TimeAwareObservation wrapper."""

import re
import warnings

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.spaces import Box, Dict, Tuple
from gymnasium.wrappers import TimeAwareObservation, TimeLimit
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_default(env_id):
    env = gym.make(env_id, disable_env_checker=True)
    wrapped_env = TimeAwareObservation(env)

    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(wrapped_env.observation_space, spaces.Box)
    assert wrapped_env.observation_space.shape[0] == env.observation_space.shape[0] + 1

    obs, info = env.reset()
    wrapped_obs, wrapped_obs_info = wrapped_env.reset()
    assert wrapped_env.timesteps == 0.0
    assert wrapped_obs[-1] == 0.0, wrapped_obs
    assert wrapped_obs.shape[0] == obs.shape[0] + 1

    wrapped_obs, _, _, _, _ = wrapped_env.step(env.action_space.sample())
    assert wrapped_env.timesteps == 1.0
    assert wrapped_obs[-1] == 1.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1

    wrapped_obs, _, _, _, _ = wrapped_env.step(env.action_space.sample())
    assert wrapped_env.timesteps == 2.0
    assert wrapped_obs[-1] == 2.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1

    wrapped_obs, wrapped_obs_info = wrapped_env.reset()
    assert wrapped_env.timesteps == 0.0
    assert wrapped_obs[-1] == 0.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1


def test_no_spec():
    env = CartPoleEnv()

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The environment must be wrapped by a TimeLimit wrapper or the spec specify a `max_episode_steps`."
        ),
    ):
        TimeAwareObservation(env)

    env = TimeLimit(env, 100)
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = TimeAwareObservation(env)

        assert env.max_timesteps == 100
    assert len(caught_warnings) == 0


def test_no_flatten():
    """Test the TimeAwareObservation wrapper without flattening the space."""
    env = GenericTestEnv(observation_space=Box(0, 1))
    wrapped_env = TimeAwareObservation(env)
    assert isinstance(wrapped_env.observation_space, Box)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert reset_obs.shape == (2,) and step_obs.shape == (2,)

    assert reset_obs in wrapped_env.observation_space
    assert step_obs in wrapped_env.observation_space


def test_with_flatten():
    """Test the flatten parameter for the TimeAwareObservation wrapper on three types of observation spaces."""
    env = GenericTestEnv(observation_space=Dict(arm_1=Box(0, 1), arm_2=Box(2, 3)))
    wrapped_env = TimeAwareObservation(env, flatten=False)
    assert isinstance(wrapped_env.observation_space, Dict)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert "time" in reset_obs and "time" in step_obs, f"{reset_obs}, {step_obs}"

    assert reset_obs in wrapped_env.observation_space
    assert step_obs in wrapped_env.observation_space

    env = GenericTestEnv(observation_space=Tuple((Box(0, 1), Box(2, 3))))
    wrapped_env = TimeAwareObservation(env, flatten=False)
    assert isinstance(wrapped_env.observation_space, Tuple)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert len(reset_obs) == 3 and len(step_obs) == 3

    assert reset_obs in wrapped_env.observation_space
    assert step_obs in wrapped_env.observation_space

    env = GenericTestEnv(observation_space=Box(0, 1))
    wrapped_env = TimeAwareObservation(env, flatten=False)
    assert isinstance(wrapped_env.observation_space, Dict)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert isinstance(reset_obs, dict) and isinstance(step_obs, dict)
    assert "obs" in reset_obs and "obs" in step_obs
    assert "time" in reset_obs and "time" in step_obs

    assert reset_obs in wrapped_env.observation_space
    assert step_obs in wrapped_env.observation_space


def test_normalize_time():
    """Test the normalize time parameter for DelayObservation wrappers."""
    env = GenericTestEnv(observation_space=Box(0, 1))
    wrapped_env = TimeAwareObservation(env, flatten=False, normalize_time=False)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert reset_obs["time"] == np.array([0], dtype=np.int32) and step_obs[
        "time"
    ] == np.array([1], dtype=np.int32)

    assert reset_obs in wrapped_env.observation_space
    assert step_obs in wrapped_env.observation_space

    env = GenericTestEnv(observation_space=Box(0, 1))
    wrapped_env = TimeAwareObservation(env, flatten=False, normalize_time=True)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert reset_obs["time"] == 0.0 and step_obs["time"] == 0.01

    assert reset_obs in wrapped_env.observation_space
    assert step_obs in wrapped_env.observation_space
