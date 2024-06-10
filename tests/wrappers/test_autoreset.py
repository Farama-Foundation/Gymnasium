"""Test suite for Autoreset wrapper."""

import numpy as np

import gymnasium as gym
from gymnasium.wrappers import Autoreset
from tests.testing_env import GenericTestEnv


def autoreset_reset_func(self: gym.Env, seed=None, options=None):
    self.count = 0
    return np.array([self.count]), {"count": self.count}


def autoreset_step_func(self: gym.Env, action: int):
    self.count += 1
    return (
        np.array([self.count]),  # Obs
        self.count > 2,  # Reward
        self.count > 2,  # Terminated
        False,  # Truncated
        {"count": self.count},  # Info
    )


def test_autoreset_wrapper_autoreset():
    """Tests the autoreset wrapper actually automatically resets correctly."""
    env = GenericTestEnv(reset_func=autoreset_reset_func, step_func=autoreset_step_func)
    env = Autoreset(env)

    obs, info = env.reset()
    assert obs == np.array([0])
    assert info == {"count": 0}

    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([1])
    assert reward == 0
    assert (terminated or truncated) is False
    assert info == {"count": 1}

    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([2])
    assert (terminated or truncated) is False
    assert reward == 0
    assert info == {"count": 2}

    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([3])
    assert (terminated or truncated) is True
    assert reward == 1
    assert info == {"count": 3}

    obs, reward, terminated, truncated, info = env.step(action)
    assert obs == np.array([0])
    assert reward == 0
    assert (terminated or truncated) is False
    assert info == {"count": 0}

    env.close()
