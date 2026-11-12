"""Test suite for ReshapeObservation wrapper."""

import re

import pytest

from gymnasium.spaces import Box
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import ReshapeObservation
from gymnasium.wrappers.vector import ReshapeObservation as VectorReshapeObservation
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import (
    check_obs,
    record_random_obs_reset,
    record_random_obs_step,
)


def test_reshape_observation_wrapper():
    """Test the ``ReshapeObservation`` wrapper."""
    env = GenericTestEnv(
        observation_space=Box(0, 1, shape=(2, 3, 2)),
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    wrapped_env = ReshapeObservation(env, (6, 2))

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == (6, 2)

    obs, _, _, _, info = wrapped_env.step(None)
    check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == (6, 2)


@pytest.mark.parametrize(
    "shape, expected_shape",
    [((-1,), (12,)), ((-1, 2), (6, 2)), ((3, -1), (3, 4)), ((2, -1, 3), (2, 2, 3))],
)
def test_reshape_observation_inferred_dimension(shape, expected_shape):
    """Test that a ``-1`` element of the shape is inferred from the original observation shape."""
    env = GenericTestEnv(
        observation_space=Box(0, 1, shape=(2, 3, 2)),
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    wrapped_env = ReshapeObservation(env, shape)
    assert wrapped_env.observation_space.shape == expected_shape
    assert wrapped_env.shape == expected_shape

    obs, info = wrapped_env.reset()
    check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == expected_shape

    obs, _, _, _, info = wrapped_env.step(None)
    check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == expected_shape


def test_vector_reshape_observation_inferred_dimension():
    """Test that the vector ``ReshapeObservation`` wrapper supports a ``-1`` shape element."""
    envs = SyncVectorEnv(
        [lambda: GenericTestEnv(observation_space=Box(0, 1, shape=(2, 3, 2)))] * 2
    )
    wrapped_envs = VectorReshapeObservation(envs, (-1, 4))
    assert wrapped_envs.single_observation_space.shape == (3, 4)
    assert wrapped_envs.observation_space.shape == (2, 3, 4)

    obs, _ = wrapped_envs.reset(seed=0)
    assert obs.shape == (2, 3, 4)
    assert obs in wrapped_envs.observation_space
    wrapped_envs.close()


@pytest.mark.parametrize(
    "shape, error_type, message",
    [
        ((5, -1), ValueError, "product of the new shape to match the original shape"),
        ((5, 2), ValueError, "product of the new shape to match the original shape"),
        ((-1, -1), ValueError, "at most one shape element to be -1"),
        ((0, -1), ValueError, "greater than 0 or -1"),
        ((-2, 6), ValueError, "greater than 0 or -1"),
        ([6, 2], TypeError, "requires shape to be a tuple"),
        ((6.0, 2), TypeError, "shape elements to be integers"),
    ],
)
def test_reshape_observation_invalid_shape(shape, error_type, message):
    """Test that invalid shapes raise an error."""
    env = GenericTestEnv(observation_space=Box(0, 1, shape=(2, 3, 2)))
    with pytest.raises(error_type, match=re.escape(message)):
        ReshapeObservation(env, shape)
