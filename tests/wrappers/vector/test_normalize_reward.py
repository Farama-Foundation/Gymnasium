"""Test suite for vector NormalizeReward wrapper."""

import numpy as np
import pytest

from gymnasium import wrappers
from gymnasium.core import ActType
from gymnasium.error import InvalidBound
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from tests.testing_env import GenericTestEnv
from tests.wrappers.vector.utils import EpisodeEndTracker, make_staggered_vector_env


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


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
def test_equivalence_with_wrapper(autoreset_mode: AutoresetMode, n_steps=50):
    """With a single sub-environment, the vector wrapper should match `NormalizeReward` within the vector env.

    Unlike `NormalizeReward`, which carries `gamma * r_T` into the next episode, the vector wrapper restarts
    the accumulated return once a sub-environment is reset, i.e., at the end of the episode for same-step autoreset
    and on `reset` for disabled autoreset. Therefore, the single-env wrapper's return is restarted at the same points.
    """

    def thunk_with_normalize():
        return wrappers.NormalizeReward(thunk())

    per_env = SyncVectorEnv([thunk_with_normalize], autoreset_mode=autoreset_mode)
    per_env.reset(seed=42)

    vec_env = SyncVectorEnv([thunk], autoreset_mode=autoreset_mode)
    vec_env = wrappers.vector.NormalizeReward(vec_env)
    vec_env.reset(seed=42)

    # The environment terminates every 10 steps
    tracker = EpisodeEndTracker(1, episode_lengths=[10])
    for _ in range(n_steps):
        action = vec_env.action_space.sample()
        _, per_env_rew, per_env_term, per_env_trunc, _ = per_env.step(action)
        _, vec_rew, vec_term, vec_trunc, _ = vec_env.step(action)

        assert np.allclose(vec_rew, per_env_rew, rtol=1e-4)
        assert np.all(vec_term == per_env_term)
        assert np.all(vec_trunc == per_env_trunc)

        dones = tracker.update(vec_term, vec_trunc)
        if autoreset_mode == AutoresetMode.DISABLED and np.any(dones):
            per_env.reset(options={"reset_mask": dones})
            vec_env.reset(options={"reset_mask": dones})
        if autoreset_mode != AutoresetMode.NEXT_STEP and dones[0]:
            per_env.envs[0].discounted_reward = np.array([0.0])

    tracker.check(n_steps)

    assert vec_env.return_rms.count == per_env.envs[0].return_rms.count
    assert np.allclose(
        vec_env.return_rms.mean, per_env.envs[0].return_rms.mean, rtol=1e-4
    )
    assert np.allclose(
        vec_env.return_rms.var, per_env.envs[0].return_rms.var, rtol=1e-4
    )
    per_env.close()
    vec_env.close()


def step_id_reward_step_func(self, action: ActType):
    self.step_id += 1
    return self.observation_space.sample(), float(self.step_id), False, False, {}


def step_id_reward_thunk():
    return GenericTestEnv(step_func=step_id_reward_step_func, reset_func=reset_func)


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
def test_staggered_accumulated_reward(
    autoreset_mode: AutoresetMode, num_envs=3, n_steps=50
):
    """With sub-environments resetting on different steps, each accumulated return matches `NormalizeReward` applied to that sub-environment.

    The single-env wrapper's return is restarted at the same points as the vector wrapper, see `test_equivalence_with_wrapper`.
    """
    per_env = make_staggered_vector_env(
        step_id_reward_thunk,
        num_envs,
        autoreset_mode,
        wrappers=(wrappers.NormalizeReward,),
    )
    vec_env = wrappers.vector.NormalizeReward(
        make_staggered_vector_env(step_id_reward_thunk, num_envs, autoreset_mode)
    )
    per_env.reset(seed=42)
    vec_env.reset(seed=42)

    tracker = EpisodeEndTracker(num_envs)
    for _ in range(n_steps):
        action = vec_env.action_space.sample()
        _, _, per_env_term, per_env_trunc, _ = per_env.step(action)
        _, _, vec_term, vec_trunc, _ = vec_env.step(action)
        assert np.all(vec_term == per_env_term)
        assert np.all(vec_trunc == per_env_trunc)

        dones = tracker.update(vec_term, vec_trunc)
        # For same-step autoreset, the accumulated return is restarted within the step,
        # while for disabled autoreset, it is restarted on `reset`
        if autoreset_mode == AutoresetMode.SAME_STEP:
            for i in np.flatnonzero(dones):
                per_env.envs[i].discounted_reward = np.array([0.0])

        for i in range(num_envs):
            assert np.isclose(
                vec_env.accumulated_reward[i], per_env.envs[i].discounted_reward[0]
            ), f"sub-environment {i}"

        if autoreset_mode == AutoresetMode.DISABLED and np.any(dones):
            per_env.reset(options={"reset_mask": dones})
            vec_env.reset(options={"reset_mask": dones})
            for i in np.flatnonzero(dones):
                per_env.envs[i].discounted_reward = np.array([0.0])
            assert np.all(vec_env.accumulated_reward[dones] == 0.0)

    tracker.check(n_steps)
    per_env.close()
    vec_env.close()


def test_partial_reset_only_clears_reset_sub_envs(n_envs=3):
    """A partial reset must only restart the accumulated return of the sub-environments being reset."""
    vec_env = wrappers.vector.NormalizeReward(
        SyncVectorEnv(
            [thunk for _ in range(n_envs)], autoreset_mode=AutoresetMode.DISABLED
        )
    )
    vec_env.reset(seed=123)
    for _ in range(3):
        vec_env.step(vec_env.action_space.sample())

    # Sets distinct accumulated rewards and episode ends to check the per sub-environment reset
    vec_env.accumulated_reward[:] = [1.0, 2.0, 3.0]
    vec_env._prev_dones[:] = [1.0, 1.0, 0.0]

    vec_env.reset(options={"reset_mask": np.array([True, False, False])})
    assert np.all(vec_env.accumulated_reward == [0.0, 2.0, 3.0])
    assert np.all(vec_env._prev_dones == [0.0, 1.0, 0.0])

    vec_env.reset(seed=123)
    assert np.all(vec_env.accumulated_reward == 0.0)
    assert np.all(vec_env._prev_dones == 0.0)
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
