"""Test suite for lambda observation wrappers."""

import numpy as np

from gymnasium.experimental.wrappers import LambdaObservationV0
from gymnasium.spaces import Box
from tests.experimental.wrappers.utils import (
    check_obs,
    record_action_as_obs_step,
    record_obs_reset,
)
from tests.testing_env import GenericTestEnv


def test_lambda_observation_wrapper():
    """Tests lambda observation that the function is applied to both the reset and step observation."""
    env = GenericTestEnv(
        reset_func=record_obs_reset, step_func=record_action_as_obs_step
    )
    wrapped_env = LambdaObservationV0(env, lambda _obs: _obs + 2, Box(2, 3))

    obs, info = wrapped_env.reset(options={"obs": np.array([0], dtype=np.float32)})
    check_obs(env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(np.array([1], dtype=np.float32))
    check_obs(env, wrapped_env, obs, info["obs"])
