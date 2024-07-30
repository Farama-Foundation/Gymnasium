"""Test suite for vector NormalizeObservation wrapper."""

import numpy as np

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


def test_normalization(
    n_envs: int = 2, convergence_steps: int = 250, testing_steps: int = 100
):
    vec_env = SyncVectorEnv([create_env for _ in range(n_envs)])
    vec_env = wrappers.vector.NormalizeObservation(vec_env)

    vec_env.reset(seed=123)
    vec_env.observation_space.seed(123)
    vec_env.action_space.seed(123)
    for _ in range(convergence_steps):
        vec_env.step(vec_env.action_space.sample())

    observations = []
    for _ in range(testing_steps):
        obs, *_ = vec_env.step(vec_env.action_space.sample())
        observations.append(obs)
    observations = np.array(observations)  # (100, 2, 3)

    mean_obs = np.mean(observations, axis=(0, 1))
    var_obs = np.var(observations, axis=(0, 1))
    assert mean_obs.shape == (3,) and var_obs.shape == (3,)

    assert np.allclose(mean_obs, np.zeros(3), atol=0.15)
    assert np.allclose(var_obs, np.ones(3), atol=0.2)


def test_wrapper_equivalence(
    n_envs: int = 3,
    n_steps: int = 250,
    mean_rtol=np.array([0.1, 0.4, 0.25]),
    var_rtol=np.array([0.15, 0.15, 0.18]),
):
    vec_env = SyncVectorEnv([create_env for _ in range(n_envs)])
    vec_env = wrappers.vector.NormalizeObservation(vec_env)

    vec_env.reset(seed=123)
    vec_env.observation_space.seed(123)
    vec_env.action_space.seed(123)
    for _ in range(n_steps):
        vec_env.step(vec_env.action_space.sample())

    env = wrappers.Autoreset(create_env())
    env = wrappers.NormalizeObservation(env)
    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(n_steps // n_envs):
        env.step(env.action_space.sample())

    assert np.allclose(env.obs_rms.mean, vec_env.obs_rms.mean, rtol=mean_rtol)
    assert np.allclose(env.obs_rms.var, vec_env.obs_rms.var, rtol=var_rtol)


def test_update_running_mean():
    env = SyncVectorEnv([create_env for _ in range(2)])
    env = wrappers.vector.NormalizeObservation(env)

    # Default value is True
    assert env.update_running_mean

    env.reset()
    for _ in range(100):
        env.step(env.action_space.sample())

    # Disable updating the running mean
    env.update_running_mean = False
    copied_rms_mean = np.copy(env.obs_rms.mean)
    copied_rms_var = np.copy(env.obs_rms.var)

    # Continue stepping through the environment and check that the running mean is not effected
    for i in range(10):
        env.step(env.action_space.sample())

    assert np.all(copied_rms_mean == env.obs_rms.mean)
    assert np.all(copied_rms_var == env.obs_rms.var)

    # Re-enable updating the running mean
    env.update_running_mean = True

    for i in range(10):
        env.step(env.action_space.sample())

    assert np.any(copied_rms_mean != env.obs_rms.mean)
    assert np.any(copied_rms_var != env.obs_rms.var)
