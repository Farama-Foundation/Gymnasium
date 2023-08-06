"""Test suite for vector NormalizeReward wrapper."""
from typing import Optional

import numpy as np

from gymnasium import wrappers
from gymnasium.core import ActType
from gymnasium.vector import SyncVectorEnv
from tests.testing_env import GenericTestEnv


def make_env():
    def reset_func(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.step_id = 0
        return self.observation_space.sample(), {}

    def step_func(self, action: ActType):
        self.step_id += 1
        done = self.step_id == 10
        print(self.step_id)
        return self.observation_space.sample(), float(done), done, False, {}

    def thunk():
        return GenericTestEnv(step_func=step_func, reset_func=reset_func)

    return thunk


def test_normalize_rew():
    env_fns = [make_env() for _ in range(8)]
    env = SyncVectorEnv(env_fns)
    env = wrappers.vector.NormalizeRewardV1(env)

    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)

    env.reset()
    forward_rets = []
    accumulated_rew = 0
    for _ in range(10):
        action = env.action_space.sample()
        _, rew, ter, tru, _ = env.step(action)
        dones = np.logical_or(ter, tru)
        accumulated_rew = accumulated_rew * 0.9 * dones + rew
        forward_rets.append(accumulated_rew)

    env.close()

    forward_rets = np.asarray(forward_rets)
    assert np.allclose(np.std(forward_rets), 1, atol=0.2)


def test_against_wrapper():
    env_fns = [make_env() for _ in range(8)]
    vec_env = SyncVectorEnv(env_fns)
    vec_env = wrappers.vector.NormalizeRewardV1(vec_env)
    vec_env.reset()
    for _ in range(100):
        action = vec_env.action_space.sample()
        vec_env.step(action)

    env = make_env()()
    env = wrappers.NormalizeRewardV1(env)
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        _, _, ter, tru, _ = env.step(action)
        if ter or tru:
            env.reset()

    rtol = 0.01
    atol = 0
    assert np.allclose(env.return_rms.var, vec_env.return_rms.var, rtol=rtol, atol=atol)
