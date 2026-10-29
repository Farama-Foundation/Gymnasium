"""Test suite for FlattenObservation wrapper."""

import numpy as np
import pytest

from gymnasium.spaces import Box, Dict, OneOf, Tuple, flatten_space
from gymnasium.wrappers import FlattenObservation
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import (
    SEED,
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


def test_flatten_observation_wrapper_oneof():
    """Tests that ``FlattenObservation`` observations of a `OneOf` space are within the wrapper's observation space."""
    env = GenericTestEnv(
        observation_space=OneOf([Box(-1, 1, shape=(2,)), Box(-1, 1, shape=(3,))]),
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    env.observation_space.seed(SEED)
    wrapped_env = FlattenObservation(env)

    assert wrapped_env.observation_space == flatten_space(env.observation_space)

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"])

    for _ in range(10):
        obs, _, _, _, info = wrapped_env.step(None)
        check_obs(env, wrapped_env, obs, info["obs"])


@pytest.mark.parametrize(
    ("space", "expected_dim"),
    [
        (Dict({}), 0),
        (Tuple(()), 0),
        (Dict(state=Box(-1, 1, shape=(2,), dtype=np.float32), optional=Dict({})), 2),
        (Tuple((Box(-1, 1, shape=(2,), dtype=np.float32), Tuple(()))), 2),
    ],
    ids=[
        "empty-dict",
        "empty-tuple",
        "dict-with-empty-group",
        "tuple-with-empty-group",
    ],
)
def test_flatten_observation_empty_composite(space, expected_dim):
    """Empty observation groups remain valid during reset and step."""
    space.seed(SEED)
    env = GenericTestEnv(
        observation_space=space,
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    wrapped_env = FlattenObservation(env)

    assert wrapped_env.observation_space.shape == (expected_dim,)
    assert wrapped_env.observation_space.dtype == np.float32
    assert wrapped_env.action_space == env.action_space

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"], strict=False)
    assert obs.shape == (expected_dim,)
    assert obs.dtype == np.float32

    obs, _, _, _, info = wrapped_env.step(None)
    check_obs(env, wrapped_env, obs, info["obs"], strict=False)
    assert obs.shape == (expected_dim,)
    assert obs.dtype == np.float32
    wrapped_env.close()
