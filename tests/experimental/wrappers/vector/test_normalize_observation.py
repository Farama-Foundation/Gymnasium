"""Test suite for NormalizeObservationV0."""
import numpy as np

from gymnasium import spaces, wrappers
from gymnasium.vector import SyncVectorEnv
from tests.testing_env import GenericTestEnv


def thunk():
    return GenericTestEnv(
        observation_space=spaces.Box(
            np.full((10,), fill_value=-100),
            np.full((10,), fill_value=10),
            dtype=np.float32,
        )
    )


def test_normalize_obs():
    env_fns = [thunk for _ in range(16)]
    env = SyncVectorEnv(env_fns)
    env = wrappers.vector.NormalizeObservationV0(env)

    # Default value is True
    assert env.update_running_mean

    obs, _ = env.reset()
    assert obs in env.observation_space
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)

    env.update_running_mean = False
    rms_var = env.obs_rms.var
    rms_mean = env.obs_rms.mean

    val_step = 25
    obs_buffer = np.empty(
        (val_step,) + env.observation_space.shape, dtype=env.observation_space.dtype
    )
    for i in range(val_step):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        obs_buffer[i] = obs

    assert np.all(rms_var == env.obs_rms.var)
    assert np.all(rms_mean == env.obs_rms.mean)
    assert np.allclose(np.std(obs_buffer, axis=0), 1, atol=1)
    assert np.allclose(np.mean(obs_buffer, axis=0), 0, atol=1)


def test_against_wrapper():
    n_envs, n_steps = 16, 100
    env_fns = [thunk for _ in range(n_envs)]
    vec_env = SyncVectorEnv(env_fns)
    vec_env = wrappers.vector.NormalizeObservationV0(vec_env)

    vec_env.reset()
    for _ in range(n_steps):
        action = vec_env.action_space.sample()
        vec_env.step(action)

    env = thunk()
    env = wrappers.NormalizeObservationV0(env)
    env.reset()
    for _ in range(n_envs * n_steps):
        action = env.action_space.sample()
        env.step(action)

    rtol = 0.07
    atol = 0
    assert np.allclose(env.obs_rms.mean, vec_env.obs_rms.mean, rtol=rtol, atol=atol)
    assert np.allclose(env.obs_rms.var, vec_env.obs_rms.var, rtol=rtol, atol=atol)
