"""Test suite for vector NormalizeReward wrapper."""

import numpy as np
import pytest

from gymnasium import wrappers
from gymnasium.core import ActType
from gymnasium.error import InvalidBound
from gymnasium.vector import AutoresetMode, SyncVectorEnv
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
    assert np.allclose(np.std(forward_rets, axis=0), 0.89, atol=0.1)


def test_against_wrapper(n_envs=3, n_steps=100, rtol=0.1, atol=0):
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


def test_equivalence_with_wrapper(n_steps=50):
    def thunk_with_normalize():
        return wrappers.NormalizeReward(thunk())

    per_env = SyncVectorEnv([thunk_with_normalize])
    per_env.reset(seed=42)
    for _ in range(n_steps):
        per_env.step(per_env.action_space.sample())

    vec_env = SyncVectorEnv([thunk])
    vec_env = wrappers.vector.NormalizeReward(vec_env)
    vec_env.reset(seed=42)
    for _ in range(n_steps):
        vec_env.step(vec_env.action_space.sample())

    assert vec_env.return_rms.count == per_env.envs[0].return_rms.count
    assert np.allclose(
        vec_env.return_rms.mean, per_env.envs[0].return_rms.mean, rtol=1e-4
    )
    assert np.allclose(
        vec_env.return_rms.var, per_env.envs[0].return_rms.var, rtol=1e-4
    )
    per_env.close()
    vec_env.close()


@pytest.mark.parametrize("gamma", [-1.0, 1.01, 2.0, 99])
def test_gamma_outside_unit_interval_is_rejected(gamma):
    """Matches the same check on the non-vector wrapper, which shares this accumulator."""
    vec_env = SyncVectorEnv([thunk])
    with pytest.raises(InvalidBound, match="`gamma` should be in the interval"):
        wrappers.vector.NormalizeReward(vec_env, gamma=gamma)
    vec_env.close()


@pytest.mark.parametrize("epsilon", [0.0, -1e-8, -1.0])
def test_non_positive_epsilon_is_rejected(epsilon):
    vec_env = SyncVectorEnv([thunk])
    with pytest.raises(InvalidBound, match="`epsilon` should be strictly positive"):
        wrappers.vector.NormalizeReward(vec_env, epsilon=epsilon)
    vec_env.close()


def test_same_step_autoreset_updates_return_rms(n_envs=2, episode_length=4, n_steps=12):
    """SAME_STEP first-after-done rewards must update return_rms and start a new return."""

    def reset_func(self, seed=None, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}

    def step_func(self, action):
        self.timestep += 1
        return (
            self.observation_space.sample(),
            1.0,
            self.timestep >= episode_length,
            False,
            {},
        )

    env = wrappers.vector.NormalizeReward(
        SyncVectorEnv(
            [
                lambda: GenericTestEnv(reset_func=reset_func, step_func=step_func)
                for _ in range(n_envs)
            ],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
    )

    env.reset(seed=123)
    for _ in range(n_steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        dones = np.logical_or(terminated, truncated)
        if np.any(dones):
            assert np.all(env.accumulated_reward[dones] == 0)

    # Every step is a real environment step under SAME_STEP.
    assert np.isclose(env.return_rms.count, n_steps * n_envs)
    env.close()
