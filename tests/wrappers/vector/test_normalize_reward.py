"""Test suite for vector NormalizeReward wrapper."""

import numpy as np

from gymnasium import wrappers
from gymnasium.core import ActType
from gymnasium.vector import SyncVectorEnv
from tests.testing_env import GenericTestEnv


def reset_func(self, seed: int | None = None, options: dict | None = None):
    self.step_id = 0
    return self.observation_space.sample(), {}


def step_func(self, action: ActType):
    self.step_id += 1
    terminated = self.step_id == 10
    return self.observation_space.sample(), float(terminated), terminated, False, {}


def thunk():
    return GenericTestEnv(step_func=step_func, reset_func=reset_func)


def test_functionality(
    n_envs=3,
    n_steps=100,
):
    env = SyncVectorEnv([thunk for _ in range(n_envs)])
    env = wrappers.vector.NormalizeReward(env)

    env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        env.step(action)

    env.reset()
    forward_rets = []
    accumulated_rew = 0
    for _ in range(n_steps):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        dones = np.logical_or(terminated, truncated)
        accumulated_rew = accumulated_rew * 0.9 * dones + reward
        forward_rets.append(accumulated_rew)

    env.close()

    forward_rets = np.asarray(forward_rets)
    assert np.allclose(np.std(forward_rets, axis=0), 1.33, atol=0.1)


def test_against_wrapper(n_envs=3, n_steps=100, rtol=0.01, atol=0):
    vec_env = SyncVectorEnv([thunk for _ in range(n_envs)])
    vec_env = wrappers.vector.NormalizeReward(vec_env)
    vec_env.reset()
    for _ in range(n_steps):
        action = vec_env.action_space.sample()
        vec_env.step(action)

    env = wrappers.Autoreset(thunk())
    env = wrappers.NormalizeReward(env)
    env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        env.step(action)

    assert np.allclose(env.return_rms.var, vec_env.return_rms.var, rtol=rtol, atol=atol)
