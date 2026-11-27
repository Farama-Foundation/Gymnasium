"""Test suite for NormalizeObservation wrapper."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces, wrappers
from gymnasium.error import InvalidBound
from gymnasium.wrappers import NormalizeObservation
from tests.testing_env import GenericTestEnv


def test_normalization(convergence_steps: int = 1000, testing_steps: int = 100):
    env = GenericTestEnv(
        observation_space=spaces.Box(
            low=np.array([0, -10, -5], dtype=np.float32),
            high=np.array([10, -5, 10], dtype=np.float32),
        )
    )
    env = wrappers.NormalizeObservation(env)

    env.reset(seed=123)
    env.observation_space.seed(123)
    env.action_space.seed(123)
    for _ in range(convergence_steps):
        env.step(env.action_space.sample())

    observations = []
    for _ in range(testing_steps):
        obs, *_ = env.step(env.action_space.sample())
        observations.append(obs)
    observations = np.array(observations)  # (100, 3)

    mean_obs = np.mean(observations, axis=0)
    var_obs = np.var(observations, axis=0)
    assert mean_obs.shape == (3,) and var_obs.shape == (3,)

    assert np.allclose(mean_obs, np.zeros(3), atol=0.15)
    assert np.allclose(var_obs, np.ones(3), atol=0.15)


def test_update_running_mean_property():
    """Tests that the property `_update_running_mean` freezes/continues the running statistics updating."""
    env = GenericTestEnv()
    wrapped_env = NormalizeObservation(env)

    # Default value is True
    assert wrapped_env.update_running_mean

    wrapped_env.reset()
    rms_var_init = wrapped_env.obs_rms.var
    rms_mean_init = wrapped_env.obs_rms.mean

    # Statistics are updated when env.step()
    wrapped_env.step(None)
    rms_var_updated = wrapped_env.obs_rms.var
    rms_mean_updated = wrapped_env.obs_rms.mean
    assert rms_var_init != rms_var_updated
    assert rms_mean_init != rms_mean_updated

    # Assure property is set
    wrapped_env.update_running_mean = False
    assert not wrapped_env.update_running_mean

    # Statistics are frozen
    wrapped_env.step(None)
    assert rms_var_updated == wrapped_env.obs_rms.var
    assert rms_mean_updated == wrapped_env.obs_rms.mean


def test_normalize_obs_with_vector():
    def thunk():
        env = gym.make("CarRacing-v3")
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.NormalizeObservation(env)
        return env

    envs = gym.vector.SyncVectorEnv([thunk for _ in range(4)])
    obs, _ = envs.reset()


@pytest.mark.parametrize("epsilon", [0.0, -1e-8, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_epsilon_is_rejected(epsilon):
    """``epsilon`` sits under a square root; once it exceeds the variance the result is NaN.

    Left unchecked, the wrapper keeps running and silently returns NaN observations
    rather than reporting that its stability parameter cannot stabilise anything.
    """
    with pytest.raises(InvalidBound, match="`epsilon` should be strictly positive"):
        NormalizeObservation(GenericTestEnv(), epsilon=epsilon)


@pytest.mark.parametrize("epsilon", [1e-8, 1e-4, 1.0])
def test_positive_epsilon_is_accepted(epsilon):
    assert NormalizeObservation(GenericTestEnv(), epsilon=epsilon).epsilon == epsilon


def test_running_stats_float64_precision():
    """Running statistics stay in float64 so updates are not rounded away at large step counts."""
    space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    env = GenericTestEnv(
        observation_space=space,
        reset_func=lambda self, seed=None, options=None: (
            np.array([100.0], dtype=np.float32),
            {},
        ),
        step_func=lambda self, action: (
            np.array([101.0], dtype=np.float32),
            0.0,
            False,
            False,
            {},
        ),
    )
    wrapped_env = NormalizeObservation(env)

    assert wrapped_env.obs_rms.mean.dtype == np.float64
    assert wrapped_env.obs_rms.var.dtype == np.float64

    obs, _ = wrapped_env.reset(seed=0)
    assert obs.dtype == np.float32

    wrapped_env.obs_rms.update(np.full((1_000_000, 1), 100.0, dtype=np.float32))
    prev_mean = float(wrapped_env.obs_rms.mean[0])

    obs, *_ = wrapped_env.step(wrapped_env.action_space.sample())
    assert obs.dtype == np.float32
    assert wrapped_env.obs_rms.mean[0] > prev_mean
