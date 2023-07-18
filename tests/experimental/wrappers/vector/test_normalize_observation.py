"""Test suite for NormalizeObservationV0."""
import numpy as np
import pytest

from gymnasium import spaces
from gymnasium.experimental.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.experimental.wrappers.vector import NormalizeObservationV0
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize("class_", [AsyncVectorEnv, SyncVectorEnv])
def test_normalize_obs(class_):
    def thunk():
        return GenericTestEnv(
            observation_space=spaces.Box(
                np.full((10,), fill_value=-100),
                np.full((10,), fill_value=10),
                dtype=np.float32,
            )
        )

    env_fns = [thunk for _ in range(16)]
    env = class_(env_fns)
    env = NormalizeObservationV0(env)

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
