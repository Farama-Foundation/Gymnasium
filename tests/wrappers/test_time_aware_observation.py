import pytest

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeAwareObservation


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_time_aware_observation(env_id):
    env = gym.make(env_id, disable_env_checker=True)
    wrapped_env = TimeAwareObservation(env)

    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(wrapped_env.observation_space, spaces.Box)
    assert wrapped_env.observation_space.shape[0] == env.observation_space.shape[0] + 1

    obs, info = env.reset()
    wrapped_obs, wrapped_obs_info = wrapped_env.reset()
    assert wrapped_env.t == 0.0
    assert wrapped_obs[-1] == 0.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1

    wrapped_obs, _, _, _, _ = wrapped_env.step(env.action_space.sample())
    assert wrapped_env.t == 1.0
    assert wrapped_obs[-1] == 1.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1

    wrapped_obs, _, _, _, _ = wrapped_env.step(env.action_space.sample())
    assert wrapped_env.t == 2.0
    assert wrapped_obs[-1] == 2.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1

    wrapped_obs, wrapped_obs_info = wrapped_env.reset()
    assert wrapped_env.t == 0.0
    assert wrapped_obs[-1] == 0.0
    assert wrapped_obs.shape[0] == obs.shape[0] + 1
