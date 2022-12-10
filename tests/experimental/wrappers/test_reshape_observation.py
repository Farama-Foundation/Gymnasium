"""Test suite for ReshapeObservationv0."""
from gymnasium.experimental.wrappers import ReshapeObservationV0
from gymnasium.spaces import Box
from tests.experimental.wrappers.utils import (
    check_obs,
    record_random_obs_reset,
    record_random_obs_step,
)
from tests.testing_env import GenericTestEnv


def test_reshape_observation_wrapper():
    """Test the ``ReshapeObservation`` wrapper."""
    env = GenericTestEnv(
        observation_space=Box(0, 1, shape=(2, 3, 2)),
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    wrapped_env = ReshapeObservationV0(env, (6, 2))

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == (6, 2)

    obs, _, _, _, info = wrapped_env.step(None)
    check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == (6, 2)
