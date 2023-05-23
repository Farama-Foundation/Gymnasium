"""Test suite for FlattenObservationV0."""
from gymnasium.experimental.wrappers import FlattenObservationV0
from gymnasium.spaces import Box, Dict
from tests.experimental.wrappers.utils import (
    check_obs,
    record_random_obs_reset,
    record_random_obs_step,
)
from tests.testing_env import GenericTestEnv


def test_flatten_observation_wrapper():
    """Tests the ``FlattenObservation`` wrapper that the observation are flattened correctly."""
    env = GenericTestEnv(
        observation_space=Dict(arm=Box(0, 1), head=Box(2, 3)),
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    wrapped_env = FlattenObservationV0(env)

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(None)
    check_obs(env, wrapped_env, obs, info["obs"])
