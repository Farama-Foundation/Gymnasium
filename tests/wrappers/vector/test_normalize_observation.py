"""Test suite for vector NormalizeObservation wrapper.."""
import numpy as np

from gymnasium import spaces, wrappers
from gymnasium.vector import SyncVectorEnv
from gymnasium.vector.utils import create_empty_array
from tests.testing_env import GenericTestEnv


def thunk():
    return GenericTestEnv(
        observation_space=spaces.Box(
            low=np.array([0, -10, -5], dtype=np.float32),
            high=np.array([10, -5, 10], dtype=np.float32),
        )
    )


def test_against_wrapper(
    n_envs=3,
    n_steps=250,
    mean_rtol=np.array([0.1, 0.4, 0.25]),
    var_rtol=np.array([0.15, 0.15, 0.18]),
):
    vec_env = SyncVectorEnv([thunk for _ in range(n_envs)])
    vec_env = wrappers.vector.NormalizeObservationV0(vec_env)

    vec_env.reset()
    for _ in range(n_steps):
        vec_env.step(vec_env.action_space.sample())

    env = wrappers.AutoresetV0(thunk())
    env = wrappers.NormalizeObservationV0(env)
    env.reset()
    for _ in range(n_envs * n_steps):
        env.step(env.action_space.sample())

    assert np.allclose(env.obs_rms.mean, vec_env.obs_rms.mean, rtol=mean_rtol)
    assert np.allclose(env.obs_rms.var, vec_env.obs_rms.var, rtol=var_rtol)


def test_update_running_mean():
    env = SyncVectorEnv([thunk for _ in range(3)])
    env = wrappers.vector.NormalizeObservationV0(env)

    # Default value is True
    assert env.update_running_mean

    obs, _ = env.reset()
    assert obs in env.observation_space
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)

    # Disable
    env.update_running_mean = False
    rms_var = np.copy(env.obs_rms.vec_var)
    rms_mean = np.copy(env.obs_rms.vec_mean)

    val_step = 25
    obs_buffer = create_empty_array(env.observation_space, val_step)
    for i in range(val_step):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        obs_buffer[i] = obs

    assert np.all(rms_var == env.obs_rms.var)
    assert np.all(rms_mean == env.obs_rms.mean)
    assert np.allclose(np.var(obs_buffer, axis=0), env.obs_rms.var, atol=1)
    assert np.allclose(np.mean(obs_buffer, axis=0), env.obs_rms.mean, atol=1)
