from typing import Optional

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import gymnasium as gym
from gymnasium.wrappers import NormalizeObservationV0, NormalizeRewardV1
from tests.testing_env import GenericTestEnv


class DummyRewardEnv(gym.Env):
    metadata = {}

    def __init__(self, return_reward_idx=0):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float64
        )
        self.returned_rewards = [0, 1, 2, 3, 4]
        self.return_reward_idx = return_reward_idx
        self.t = self.return_reward_idx

    def step(self, action):
        self.t += 1
        return (
            np.array([self.t]),
            self.t,
            self.t == len(self.returned_rewards),
            False,
            {},
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = self.return_reward_idx
        return np.array([self.t]), {}


def make_env(return_reward_idx):
    def thunk():
        env = DummyRewardEnv(return_reward_idx)
        return env

    return thunk


def test_normalize_observation():
    """Tests that the property `_update_running_mean` freezes/continues the running statistics updating."""
    env = GenericTestEnv()
    wrapped_env = NormalizeObservationV0(env)

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


def test_normalize_reset_info():
    env = DummyRewardEnv(return_reward_idx=0)
    env = NormalizeObservationV0(env)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


def test_normalize_return():
    env = DummyRewardEnv(return_reward_idx=0)
    env = NormalizeRewardV1(env)
    env.reset()
    env.step(env.action_space.sample())
    assert_almost_equal(
        env.return_rms.mean,
        np.mean([1]),  # [first return]
        decimal=4,
    )
    env.step(env.action_space.sample())
    assert_almost_equal(
        env.return_rms.mean,
        np.mean([2 + env.gamma * 1, 1]),  # [second return, first return]
        decimal=4,
    )


def test_normalize_observation_vector_env():
    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs.reset()
    obs, reward, _, _, _ = envs.step(envs.action_space.sample())
    np.testing.assert_almost_equal(obs, np.array([[1], [2]]), decimal=4)
    np.testing.assert_almost_equal(reward, np.array([1, 2]), decimal=4)

    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs = NormalizeObservationV0(envs)
    envs.reset()
    assert_almost_equal(
        envs.obs_rms.mean,
        np.mean([0.5]),  # the mean of first observations [[0, 1]]
        decimal=4,
    )
    obs, reward, _, _, _ = envs.step(envs.action_space.sample())
    assert_almost_equal(
        envs.obs_rms.mean,
        np.mean([1.0]),  # the mean of first and second observations [[0, 1], [1, 2]]
        decimal=4,
    )


@pytest.mark.skip(reason="wrappers.vector.NormalizeReward is not yet implemented")
def test_normalize_return_vector_env():
    env_fns = [make_env(0), make_env(1)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    envs = NormalizeRewardV1(envs)
    envs.reset()
    obs, reward, _, _, _ = envs.step(envs.action_space.sample())
    assert_almost_equal(
        envs.return_rms.mean,
        np.mean([1.5]),  # the mean of first returns [[1, 2]]
        decimal=4,
    )
    obs, reward, _, _, _ = envs.step(envs.action_space.sample())
    assert_almost_equal(
        envs.return_rms.mean,
        np.mean(
            [[1, 2], [2 + envs.gamma * 1, 3 + envs.gamma * 2]]
        ),  # the mean of first and second returns [[1, 2], [2 + envs.gamma * 1, 3 + envs.gamma * 2]]
        decimal=4,
    )
