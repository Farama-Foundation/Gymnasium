"""Test suite for RescaleObservationV0."""
import numpy as np

from gymnasium.experimental.wrappers import RescaleObservationV0
from gymnasium.spaces import Box
from tests.experimental.wrappers.utils import (
    check_obs,
    record_action_as_obs_step,
    record_obs_reset,
)
from tests.testing_env import GenericTestEnv


def test_rescale_observation():
    """Test the ``RescaleObservation`` wrapper."""
    env = GenericTestEnv(
        observation_space=Box(
            np.array([0, 1], dtype=np.float32), np.array([1, 3], dtype=np.float32)
        ),
        reset_func=record_obs_reset,
        step_func=record_action_as_obs_step,
    )
    wrapped_env = RescaleObservationV0(
        env,
        min_obs=np.array([-5, 0], dtype=np.float32),
        max_obs=np.array([5, 1], dtype=np.float32),
    )
    assert wrapped_env.observation_space == Box(
        np.array([-5, 0], dtype=np.float32), np.array([5, 1], dtype=np.float32)
    )

    for sample_obs, expected_obs in (
        (
            np.array([0.5, 2.0], dtype=np.float32),
            np.array([0.0, 0.5], dtype=np.float32),
        ),
        (
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([-5.0, 0.0], dtype=np.float32),
        ),
        (
            np.array([1.0, 3.0], dtype=np.float32),
            np.array([5.0, 1.0], dtype=np.float32),
        ),
    ):
        assert sample_obs in env.observation_space
        assert expected_obs in wrapped_env.observation_space

        obs, info = wrapped_env.reset(options={"obs": sample_obs})
        assert np.all(obs == expected_obs)
        check_obs(env, wrapped_env, obs, info["obs"], strict=False)

        obs, _, _, _, info = wrapped_env.step(sample_obs)
        assert np.all(obs == expected_obs)
        check_obs(env, wrapped_env, obs, info["obs"], strict=False)
