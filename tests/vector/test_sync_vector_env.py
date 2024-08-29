"""Test the `SyncVectorEnv` implementation."""

import re

import numpy as np
import pytest

from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple
from gymnasium.vector import SyncVectorEnv
from tests.envs.utils import all_testing_env_specs
from tests.vector.testing_utils import (
    CustomSpace,
    assert_rng_equal,
    make_custom_space_env,
    make_env,
)


def test_create_sync_vector_env():
    """Tests creating the sync vector environment."""
    env_fns = [make_env("FrozenLake-v1", i) for i in range(8)]
    env = SyncVectorEnv(env_fns)
    env.close()

    assert env.num_envs == 8


def test_reset_sync_vector_env():
    """Tests sync vector `reset` function."""
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]
    env = SyncVectorEnv(env_fns)
    observations, infos = env.reset()
    env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape

    del observations


@pytest.mark.parametrize("use_single_action_space", [True, False])
def test_step_sync_vector_env(use_single_action_space):
    """Test sync vector `steps` function."""
    env = SyncVectorEnv([make_env("FrozenLake-v1", i) for i in range(8)])
    env.reset()

    assert isinstance(env.single_action_space, Discrete)
    assert isinstance(env.action_space, MultiDiscrete)

    if use_single_action_space:
        actions = [env.single_action_space.sample() for _ in range(8)]
    else:
        actions = env.action_space.sample()
    observations, rewards, terminations, truncations, _ = env.step(actions)

    env.close()

    assert isinstance(env.observation_space, MultiDiscrete)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape

    assert isinstance(rewards, np.ndarray)
    assert isinstance(rewards[0], (float, np.floating))
    assert rewards.ndim == 1
    assert rewards.size == 8

    assert isinstance(terminations, np.ndarray)
    assert terminations.dtype == np.bool_
    assert terminations.ndim == 1
    assert terminations.size == 8

    assert isinstance(truncations, np.ndarray)
    assert truncations.dtype == np.bool_
    assert truncations.ndim == 1
    assert truncations.size == 8


def test_render_sync_vector():
    envs = SyncVectorEnv(
        [make_env("CartPole-v1", i, render_mode="rgb_array") for i in range(3)]
    )
    assert envs.render_mode == "rgb_array"

    envs.reset()
    rendered_frames = envs.render()
    assert isinstance(rendered_frames, tuple)
    assert len(rendered_frames) == envs.num_envs
    assert all(isinstance(frame, np.ndarray) for frame in rendered_frames)

    envs = SyncVectorEnv([make_env("CartPole-v1", i) for i in range(3)])
    assert envs.render_mode is None


def test_call_sync_vector_env():
    """Test sync vector `call` on sub-environments."""
    env_fns = [
        make_env("CartPole-v1", i, render_mode="rgb_array_list") for i in range(4)
    ]

    env = SyncVectorEnv(env_fns)
    _ = env.reset()
    images = env.call("render")
    gravity = env.call("gravity")

    env.close()

    assert isinstance(images, tuple)
    assert len(images) == 4
    for i in range(4):
        assert len(images[i]) == 1
        assert isinstance(images[i][0], np.ndarray)

    assert isinstance(gravity, tuple)
    assert len(gravity) == 4
    for i in range(4):
        assert isinstance(gravity[i], float)
        assert gravity[i] == 9.8


def test_set_attr_sync_vector_env():
    """Test sync vector `set_attr` function."""
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]

    env = SyncVectorEnv(env_fns)
    env.set_attr("gravity", [9.81, 3.72, 8.87, 1.62])
    gravity = env.get_attr("gravity")
    assert gravity == (9.81, 3.72, 8.87, 1.62)

    env.close()


def test_check_spaces_sync_vector_env():
    """Tests the sync vector `check_spaces` function."""
    # CartPole-v1 - observation_space: Box(4,), action_space: Discrete(2)
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]
    # FrozenLake-v1 - Discrete(16), action_space: Discrete(4)
    env_fns[1] = make_env("FrozenLake-v1", 1)
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "SyncVectorEnv(..., observation_mode='same') however the sub-environments observation spaces are not equivalent."
        ),
    ):
        env = SyncVectorEnv(env_fns)
        env.close()


def test_custom_space_sync_vector_env():
    """Test the use of custom spaces with sync vector environment."""
    env_fns = [make_custom_space_env(i) for i in range(4)]

    env = SyncVectorEnv(env_fns)
    reset_observations, infos = env.reset()

    assert isinstance(env.single_action_space, CustomSpace)
    assert isinstance(env.action_space, Tuple)

    actions = ("action-2", "action-3", "action-5", "action-7")
    step_observations, _, _, _, _ = env.step(actions)

    env.close()

    assert isinstance(env.single_observation_space, CustomSpace)
    assert isinstance(env.observation_space, Tuple)

    assert isinstance(reset_observations, tuple)
    assert reset_observations == ("reset", "reset", "reset", "reset")

    assert isinstance(step_observations, tuple)
    assert step_observations == (
        "step(action-2)",
        "step(action-3)",
        "step(action-5)",
        "step(action-7)",
    )


def test_sync_vector_env_seed():
    """Test seeding for sync vector environments."""
    env = make_env("BipedalWalker-v3", seed=123)()
    sync_vector_env = SyncVectorEnv([make_env("BipedalWalker-v3", seed=123)])

    assert_rng_equal(env.action_space.np_random, sync_vector_env.action_space.np_random)
    for _ in range(100):
        env_action = env.action_space.sample()
        vector_action = sync_vector_env.action_space.sample()
        assert np.all(env_action == vector_action)

    env.close()


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_sync_vector_determinism(spec: EnvSpec, seed: int = 123, n: int = 3):
    """Check that for all environments, the sync vector envs produce the same action samples using the same seeds."""
    env_1 = SyncVectorEnv([make_env(spec.id, seed=seed) for _ in range(n)])
    env_2 = SyncVectorEnv([make_env(spec.id, seed=seed) for _ in range(n)])
    assert_rng_equal(env_1.action_space.np_random, env_2.action_space.np_random)

    for _ in range(100):
        env_1_samples = env_1.action_space.sample()
        env_2_samples = env_2.action_space.sample()
        assert np.all(env_1_samples == env_2_samples)

    env_1.close()
    env_2.close()
