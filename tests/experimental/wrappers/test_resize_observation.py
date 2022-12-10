"""Test suite for ResizeObservationV0."""
import numpy as np

from gymnasium.experimental.wrappers import ResizeObservationV0
from gymnasium.spaces import Box
from tests.experimental.wrappers.utils import (
    check_obs,
    record_random_obs_reset,
    record_random_obs_step,
)
from tests.testing_env import GenericTestEnv


def test_resize_observation_wrapper():
    """Test the ``ResizeObservation`` that the observation has changed size."""
    env = GenericTestEnv(
        observation_space=Box(0, 255, shape=(60, 60, 3), dtype=np.uint8),
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    wrapped_env = ResizeObservationV0(env, (25, 25))

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(None)
    check_obs(env, wrapped_env, obs, info["obs"])
