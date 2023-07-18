"""Test suite for NormalizeRewardV1."""
from typing import Optional

import numpy as np
import pytest

from gymnasium.core import ActType
from gymnasium.experimental.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.experimental.wrappers.vector import NormalizeRewardV1
from tests.testing_env import GenericTestEnv


def make_env():
    def reset_func(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.step_id = 0
        return self.observation_space.sample(), {}

    def step_func(self, action: ActType):
        self.step_id += 1
        done = self.step_id == 10
        return self.observation_space.sample(), float(done), done, False, {}

    def thunk():
        return GenericTestEnv(step_func=step_func, reset_func=reset_func)

    return thunk


@pytest.mark.parametrize("class_", [AsyncVectorEnv, SyncVectorEnv])
def test_normalize_rew(class_):
    env_fns = [make_env() for _ in range(8)]
    env = class_(env_fns)
    env = NormalizeRewardV1(env)

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
