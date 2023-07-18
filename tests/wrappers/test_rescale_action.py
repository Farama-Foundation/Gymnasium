import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import RescaleActionV0


def test_rescale_action():
    env = gym.make("CartPole-v1", disable_env_checker=True)
    with pytest.raises(AssertionError):
        RescaleActionV0(env, -1, 1)
    del env

    env = gym.make("Pendulum-v1", disable_env_checker=True)
    wrapped_env = RescaleActionV0(
        gym.make("Pendulum-v1", disable_env_checker=True), -1, 1
    )

    seed = 0

    obs, info = env.reset(seed=seed)
    wrapped_obs, wrapped_obs_info = wrapped_env.reset(seed=seed)
    assert np.allclose(obs, wrapped_obs)

    obs, reward, _, _, _ = env.step(np.array([1.5], dtype=np.float32))
    # with pytest.raises(AssertionError):
    #     wrapped_env.step([1.5])
    wrapped_obs, wrapped_reward, _, _, _ = wrapped_env.step(
        np.array([0.75], dtype=np.float32)
    )

    assert np.allclose(obs, wrapped_obs)
    assert np.allclose(reward, wrapped_reward)
