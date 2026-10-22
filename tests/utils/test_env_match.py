"""Tests for `gymnasium.utils.env_match.check_environments_match`."""

import numpy as np
import pytest

from gymnasium import spaces
from gymnasium.utils.env_match import check_environments_match
from tests.testing_env import GenericTestEnv


def _fixed_obs_reset_func(self, *, seed=None, options=None):
    """Reset function that returns a constant observation shared between test envs."""
    super(GenericTestEnv, self).reset(seed=seed)
    return np.array([0.5, 0.5], dtype=np.float32), {}


def _fixed_obs_step_func(self, action):
    """Step function that returns a constant observation shared between test envs."""
    return np.array([0.5, 0.5], dtype=np.float32), 0.0, False, False, {}


def _make_env(observation_space):
    """Create a test env with a fixed observation and the given observation space."""
    return GenericTestEnv(
        action_space=spaces.Discrete(2),
        observation_space=observation_space,
        reset_func=_fixed_obs_reset_func,
        step_func=_fixed_obs_step_func,
    )


def test_matching_environments():
    """Two identical environments pass the check."""
    env_a = _make_env(spaces.Box(0, 1, (2,), dtype=np.float32))
    env_b = _make_env(spaces.Box(0, 1, (2,), dtype=np.float32))
    check_environments_match(env_a, env_b, num_steps=5)


def test_mismatched_observation_space():
    """Environments with different observation spaces are rejected, even when the observations are equal."""
    env_a = _make_env(spaces.Box(0, 1, (2,), dtype=np.float32))
    env_b = _make_env(spaces.Box(0, 2, (2,), dtype=np.float32))
    with pytest.raises(AssertionError):
        check_environments_match(env_a, env_b, num_steps=0)


def test_mismatched_observation_space_skip_obs():
    """`skip_obs=True` skips the observation space equivalence check."""
    env_a = _make_env(spaces.Box(0, 1, (2,), dtype=np.float32))
    env_b = _make_env(spaces.Box(0, 2, (2,), dtype=np.float32))
    check_environments_match(env_a, env_b, num_steps=0, skip_obs=True)


def test_mismatched_action_space():
    """Environments with different action spaces are rejected."""
    env_a = GenericTestEnv(
        action_space=spaces.Discrete(2),
        reset_func=_fixed_obs_reset_func,
    )
    env_b = GenericTestEnv(
        action_space=spaces.Discrete(3),
        reset_func=_fixed_obs_reset_func,
    )
    with pytest.raises(AssertionError):
        check_environments_match(env_a, env_b, num_steps=0)
