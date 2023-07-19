"""Test suite for ResizeObservationV0."""
from __future__ import annotations

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.experimental.wrappers import ResizeObservationV0
from gymnasium.spaces import Box
from tests.experimental.wrappers.utils import (
    check_obs,
    record_random_obs_reset,
    record_random_obs_step,
)
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize(
    "env",
    (
        GenericTestEnv(
            observation_space=Box(0, 255, shape=(60, 60, 3), dtype=np.uint8),
            reset_func=record_random_obs_reset,
            step_func=record_random_obs_step,
        ),
        GenericTestEnv(
            observation_space=Box(0, 255, shape=(60, 60), dtype=np.uint8),
            reset_func=record_random_obs_reset,
            step_func=record_random_obs_step,
        ),
    ),
)
def test_resize_observation_wrapper(env):
    """Test the ``ResizeObservation`` that the observation has changed size."""

    wrapped_env = ResizeObservationV0(env, (25, 25))
    assert isinstance(wrapped_env.observation_space, Box)
    assert wrapped_env.observation_space.shape[:2] == (25, 25)

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(None)
    check_obs(env, wrapped_env, obs, info["obs"])


@pytest.mark.parametrize("shape", ((10, 10), (20, 20), (60, 60), (100, 100)))
def test_resize_shapes(shape: tuple[int, int]):
    env = ResizeObservationV0(gym.make("CarRacing-v2"), shape)
    assert env.observation_space == Box(
        low=0, high=255, shape=shape + (3,), dtype=np.uint8
    )

    obs, info = env.reset()
    assert obs in env.observation_space
    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs in env.observation_space
