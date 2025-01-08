"""Test suite for vector TransformAction wrapper."""

import numpy as np

from gymnasium import spaces, wrappers
from gymnasium.vector import SyncVectorEnv
from tests.testing_env import GenericTestEnv


def create_env():
    return GenericTestEnv(
        action_space=spaces.Box(
            low=np.array([0, -10, -5], dtype=np.float32),
            high=np.array([10, -5, 10], dtype=np.float32),
        )
    )


def test_observation_space_from_single_observation_space(
    n_envs: int = 5,
):
    vec_env = SyncVectorEnv([create_env for _ in range(n_envs)])
    vec_env = wrappers.vector.TransformAction(
        vec_env,
        func=lambda x: x + 100,
        single_action_space=spaces.Box(
            low=np.array([0, -10, -5], dtype=np.float32) + 100,
            high=np.array([10, -5, 10], dtype=np.float32) + 100,
        ),
    )

    # Check action space
    assert isinstance(vec_env.action_space, spaces.Box)
    assert vec_env.action_space.shape == (n_envs, 3)
    assert vec_env.action_space.dtype == np.float32
    assert (
        vec_env.action_space.low == np.array([[100, 90, 95]] * n_envs, dtype=np.float32)
    ).all()
    assert (
        vec_env.action_space.high
        == np.array([[110, 95, 110]] * n_envs, dtype=np.float32)
    ).all()

    # Check single action space
    assert isinstance(vec_env.single_action_space, spaces.Box)
    assert vec_env.single_action_space.shape == (3,)
    assert vec_env.single_action_space.dtype == np.float32
    assert (
        vec_env.single_action_space.low == np.array([100, 90, 95], dtype=np.float32)
    ).all()
    assert (
        vec_env.single_action_space.high == np.array([110, 95, 110], dtype=np.float32)
    ).all()
