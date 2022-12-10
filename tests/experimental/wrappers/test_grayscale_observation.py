"""Test suite for GrayscaleObservationV0."""
import numpy as np

from gymnasium.experimental.wrappers import GrayscaleObservationV0
from gymnasium.spaces import Box
from tests.experimental.wrappers.utils import (
    check_obs,
    record_random_obs_reset,
    record_random_obs_step,
)
from tests.testing_env import GenericTestEnv


def test_grayscale_observation_wrapper():
    """Tests the ``GrayscaleObservation`` that the observation is grayscale."""
    env = GenericTestEnv(
        observation_space=Box(0, 255, shape=(25, 25, 3), dtype=np.uint8),
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    wrapped_env = GrayscaleObservationV0(env)

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == (25, 25)

    obs, _, _, _, info = wrapped_env.step(None)
    check_obs(env, wrapped_env, obs, info["obs"])

    # Keep_dim
    wrapped_env = GrayscaleObservationV0(env, keep_dim=True)

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == (25, 25, 1)

    obs, _, _, _, info = wrapped_env.step(None)
    check_obs(env, wrapped_env, obs, info["obs"])
