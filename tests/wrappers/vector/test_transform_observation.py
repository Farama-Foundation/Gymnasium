"""Test suite for vector TransformObservation wrapper."""

import numpy as np
import pytest

from gymnasium import spaces, wrappers
from gymnasium.vector import SyncVectorEnv
from tests.testing_env import GenericTestEnv


def create_env():
    return GenericTestEnv(
        observation_space=spaces.Box(
            low=np.array([0, -10, -5], dtype=np.float32),
            high=np.array([10, -5, 10], dtype=np.float32),
        )
    )


def test_transform(n_envs: int = 2):
    vec_env = SyncVectorEnv([create_env for _ in range(n_envs)])
    vec_env = wrappers.vector.TransformObservation(
        vec_env,
        func=lambda x: x + 100,
        single_observation_space=spaces.Box(
            low=np.array([0, -10, -5], dtype=np.float32),
            high=np.array([10, -5, 10], dtype=np.float32),
        ),
    )

    obs, _ = vec_env.reset(seed=123)
    vec_env.observation_space.seed(123)
    vec_env.action_space.seed(123)

    assert (obs >= np.array([100, 90, 95], dtype=np.float32)).all()
    assert (obs <= np.array([110, 95, 110], dtype=np.float32)).all()

    obs, *_ = vec_env.step(vec_env.action_space.sample())

    assert (obs >= np.array([100, 90, 95], dtype=np.float32)).all()
    assert (obs <= np.array([110, 95, 110], dtype=np.float32)).all()


def test_observation_space_from_single_observation_space(
    n_envs: int = 5,
):
    vec_env = SyncVectorEnv([create_env for _ in range(n_envs)])
    vec_env = wrappers.vector.TransformObservation(
        vec_env,
        func=lambda x: x + 100,
        single_observation_space=spaces.Box(
            low=np.array([0, -10, -5], dtype=np.float32) + 100,
            high=np.array([10, -5, 10], dtype=np.float32) + 100,
        ),
    )

    # Check observation space
    assert isinstance(vec_env.observation_space, spaces.Box)
    assert vec_env.observation_space.shape == (n_envs, 3)
    assert vec_env.observation_space.dtype == np.float32
    assert (
        vec_env.observation_space.low
        == np.array([[100, 90, 95]] * n_envs, dtype=np.float32)
    ).all()
    assert (
        vec_env.observation_space.high
        == np.array([[110, 95, 110]] * n_envs, dtype=np.float32)
    ).all()

    # Check single observation space
    assert isinstance(vec_env.single_observation_space, spaces.Box)
    assert vec_env.single_observation_space.shape == (3,)
    assert vec_env.single_observation_space.dtype == np.float32
    assert (
        vec_env.single_observation_space.low
        == np.array([100, 90, 95], dtype=np.float32)
    ).all()
    assert (
        vec_env.single_observation_space.high
        == np.array([110, 95, 110], dtype=np.float32)
    ).all()


def test_warning_on_mismatched_single_observation_space(
    n_envs: int = 2,
):
    vec_env = SyncVectorEnv([create_env for _ in range(n_envs)])
    # We only specify observation_space without single_observation_space, so single_observation_space inherits its value from the wrapped env which would not match. This mismatch should give us a warning.
    with pytest.warns(
        Warning,
        match=r"the observation space and the batched single observation space don't match as expected",
    ):
        vec_env = wrappers.vector.TransformObservation(
            vec_env,
            func=lambda x: x + 100,
            observation_space=spaces.Box(
                low=np.array([[0, -10, -5]] * n_envs, dtype=np.float32) + 100,
                high=np.array([[10, -5, 10]] * n_envs, dtype=np.float32) + 100,
            ),
        )
