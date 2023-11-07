"""Test suite for NormalizeReward wrapper."""

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType
from gymnasium.wrappers import NormalizeReward
from tests.testing_env import GenericTestEnv


def constant_reward_step_func(self, action: ActType):
    return self.observation_space.sample(), 1.0, False, False, {}


def test_running_mean_normalize_reward_wrapper():
    """Tests that the property `_update_running_mean` freezes/continues the running statistics updating."""
    env = GenericTestEnv(step_func=constant_reward_step_func)
    wrapped_env = NormalizeReward(env)

    # Default value is True
    assert wrapped_env.update_running_mean

    wrapped_env.reset()
    rms_var_init = wrapped_env.return_rms.var
    rms_mean_init = wrapped_env.return_rms.mean

    # Statistics are updated when env.step()
    wrapped_env.step(None)
    rms_var_updated = wrapped_env.return_rms.var
    rms_mean_updated = wrapped_env.return_rms.mean
    assert rms_var_init != rms_var_updated
    assert rms_mean_init != rms_mean_updated

    # Assure property is set
    wrapped_env.update_running_mean = False
    assert not wrapped_env.update_running_mean

    # Statistics are frozen
    wrapped_env.step(None)
    assert rms_var_updated == wrapped_env.return_rms.var
    assert rms_mean_updated == wrapped_env.return_rms.mean


def test_normalize_reward_wrapper():
    """Tests that the NormalizeReward does not throw an error."""
    # TODO: Functional correctness should be tested
    env = GenericTestEnv(step_func=constant_reward_step_func)
    wrapped_env = NormalizeReward(env)
    wrapped_env.reset()
    _, reward, _, _, _ = wrapped_env.step(None)
    assert np.ndim(reward) == 0
    env.close()


def reward_reset_func(self: gym.Env, seed=None, options=None):
    self.rewards = [0, 1, 2, 3, 4]
    reward = self.rewards.pop(0)
    return np.array([reward]), {"reward": reward}


def reward_step_func(self: gym.Env, action):
    reward = self.rewards.pop(0)
    return np.array([reward]), reward, len(self.rewards) == 0, False, {"reward": reward}


def test_normalize_return():
    env = GenericTestEnv(reset_func=reward_reset_func, step_func=reward_step_func)
    env = NormalizeReward(env)
    env.reset()

    env.step(env.action_space.sample())
    np.testing.assert_almost_equal(
        env.return_rms.mean,
        np.mean([1]),  # [first return]
        decimal=4,
    )

    env.step(env.action_space.sample())
    np.testing.assert_almost_equal(
        env.return_rms.mean,
        np.mean([2 + 1 * env.gamma, 1]),  # [second return, first return]
        decimal=4,
    )
