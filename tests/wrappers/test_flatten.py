"""Tests for the flatten observation wrapper."""

from collections import OrderedDict

import pytest

from gymnasium.spaces import Box, Dict, unflatten
from gymnasium.wrappers import FlattenObservation
from tests.generic_test_env import GenericTestEnv

OBSERVATION_SPACES = (
    Dict(
        OrderedDict(
            [
                ("key1", Box(shape=(2, 3), low=0, high=0)),
                ("key2", Box(shape=(1,), low=1, high=1)),
                ("key3", Box(shape=(2,), low=2, high=2)),
            ]
        )
    ),
    Dict(
        OrderedDict(
            [
                ("key2", Box(shape=(1,), low=0, high=0)),
                ("key3", Box(shape=(2,), low=1, high=1)),
                ("key1", Box(shape=(2, 3), low=2, high=2)),
            ]
        )
    ),
    Dict(
        {
            "key1": Box(shape=(2, 3), low=-1, high=1),
            "key2": Box(shape=(1,), low=-1, high=1),
            "key3": Box(shape=(2,), low=-1, high=1),
        }
    ),
)


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
def test_flattened_environment(observation_space):
    env = GenericTestEnv(observation_space=observation_space)
    flattened_env = FlattenObservation(env)
    flattened_obs, info = flattened_env.reset()

    assert flattened_obs in flattened_env.observation_space
    assert flattened_obs not in env.observation_space

    unflattened_obs = unflatten(env.observation_space, flattened_obs)
    assert unflattened_obs in env.observation_space
    assert unflattened_obs not in flattened_env.observation_space
