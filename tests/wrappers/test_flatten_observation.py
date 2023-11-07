"""Test suite for FlattenObservation wrapper."""

from gymnasium.spaces import Box, Dict, flatten_space
from gymnasium.wrappers import FlattenObservation
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import (
    check_obs,
    record_random_obs_reset,
    record_random_obs_step,
)


def test_flatten_observation_wrapper():
    """Tests the ``FlattenObservation`` wrapper that the observation are flattened correctly."""
    env = GenericTestEnv(
        observation_space=Dict(arm=Box(0, 1), head=Box(2, 3)),
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    wrapped_env = FlattenObservation(env)

    assert wrapped_env.observation_space == flatten_space(env.observation_space)
    assert wrapped_env.action_space == env.action_space

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(None)
    check_obs(env, wrapped_env, obs, info["obs"])
