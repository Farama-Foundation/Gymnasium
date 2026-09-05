"""Test suite for NormalizeReward wrapper."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.core import ActType
from gymnasium.error import InvalidBound
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


@pytest.mark.parametrize("gamma", [-1.0, -0.01, 1.01, 2.0, 99])
def test_gamma_outside_unit_interval_is_rejected(gamma):
    """The accumulator is multiplied by ``gamma`` every step, so a value outside [0, 1] diverges.

    Left unchecked, ``gamma=99`` (a plausible typo for ``0.99``) drives the accumulator
    to infinity and every normalized reward to NaN, without raising anything.
    """
    with pytest.raises(InvalidBound, match="`gamma` should be in the interval"):
        NormalizeReward(GenericTestEnv(), gamma=gamma)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.99, 1.0])
def test_gamma_inside_unit_interval_is_accepted(gamma):
    """Both endpoints are meaningful: 0 keeps only the immediate reward, 1 is undiscounted."""
    assert NormalizeReward(GenericTestEnv(), gamma=gamma).gamma == gamma


@pytest.mark.parametrize("epsilon", [0.0, -1e-8, -1.0])
def test_non_positive_epsilon_is_rejected(epsilon):
    """``epsilon`` sits under a square root; once it exceeds the variance the result is NaN."""
    with pytest.raises(InvalidBound, match="`epsilon` should be strictly positive"):
        NormalizeReward(GenericTestEnv(), epsilon=epsilon)
