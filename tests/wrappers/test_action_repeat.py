"""Test suite for ActionRepeat wrapper."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.wrappers.stateful_action import ActionRepeat
from tests.testing_env import GenericTestEnv


def _counting_step(self, action):
    """Step function that returns an incrementing reward and the action as obs."""
    self._step_count = getattr(self, "_step_count", 0) + 1
    return action, float(self._step_count), False, False, {"step": self._step_count}


def _terminating_step(self, action):
    """Step function that terminates after 3 steps."""
    self._step_count = getattr(self, "_step_count", 0) + 1
    terminated = self._step_count >= 3
    return action, 1.0, terminated, False, {"step": self._step_count}


def test_action_repeat_exact_steps():
    """Test that exactly num_repeats inner steps are taken using GenericTestEnv."""
    env = ActionRepeat(
        GenericTestEnv(
            step_func=_counting_step,
            action_space=Discrete(5),
            observation_space=Discrete(5),
        ),
        num_repeats=3,
    )
    env.reset()
    obs, reward, _, _, info = env.step(2)
    # Rewards: 1.0 + 2.0 + 3.0 = 6.0 (proving exactly 3 inner steps)
    assert reward == 6.0
    assert info["step"] == 3  # info from last inner step
    assert obs == 2  # action passed through as observation


def test_action_repeat_reward_accumulation():
    """Test that rewards from repeated steps are summed correctly."""
    env = gym.make("CartPole-v1")

    # Without wrapper: step 4 times manually
    env.reset(seed=42)
    total_reward_manual = 0.0
    for _ in range(4):
        obs_manual, reward, term, trunc, _ = env.step(1)
        total_reward_manual += reward
        if term or trunc:
            break

    # With wrapper: single step should accumulate 4 rewards
    env2 = gym.make("CartPole-v1")
    wrapped_env = ActionRepeat(env2, num_repeats=4)
    wrapped_env.reset(seed=42)
    obs_wrapped, reward_wrapped, _, _, _ = wrapped_env.step(1)

    assert reward_wrapped == total_reward_manual
    assert np.array_equal(obs_wrapped, obs_manual)


def test_action_repeat_early_termination():
    """Test that repetition stops early on termination."""
    env = ActionRepeat(
        GenericTestEnv(
            step_func=_terminating_step,
            action_space=Discrete(3),
            observation_space=Discrete(3),
        ),
        num_repeats=10,
    )
    env.reset()
    _, reward, terminated, _, info = env.step(0)
    # Environment terminates at step 3, so only 3 inner steps taken
    assert terminated is True
    assert reward == 3.0  # 1.0 * 3 steps
    assert info["step"] == 3


def test_action_repeat_info_propagation():
    """Test that info from the last inner step is returned."""
    env = ActionRepeat(
        GenericTestEnv(
            step_func=_counting_step,
            action_space=Discrete(5),
            observation_space=Discrete(5),
        ),
        num_repeats=5,
    )
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["step"] == 5  # info from the 5th (last) inner step


def test_action_repeat_num_repeats_one():
    """Test that num_repeats=1 behaves identically to the unwrapped environment."""
    env1 = gym.make("CartPole-v1")
    env2 = gym.make("CartPole-v1")
    wrapped_env = ActionRepeat(env2, num_repeats=1)

    env1.reset(seed=42)
    wrapped_env.reset(seed=42)

    for _ in range(10):
        action = env1.action_space.sample()
        obs1, r1, term1, trunc1, _ = env1.step(action)
        obs2, r2, term2, trunc2, _ = wrapped_env.step(action)
        assert r1 == r2
        assert term1 == term2
        assert trunc1 == trunc2
        assert np.array_equal(obs1, obs2)
        if term1 or trunc1:
            break


def test_action_repeat_observation_space():
    """Test that observation and action spaces are preserved."""
    env = gym.make("CartPole-v1")
    wrapped_env = ActionRepeat(env, num_repeats=4)

    assert wrapped_env.observation_space == env.observation_space
    assert wrapped_env.action_space == env.action_space

    obs, _ = wrapped_env.reset(seed=123)
    assert obs in wrapped_env.observation_space

    for _ in range(5):
        obs, _, term, trunc, _ = wrapped_env.step(wrapped_env.action_space.sample())
        assert obs in wrapped_env.observation_space
        if term or trunc:
            obs, _ = wrapped_env.reset()


@pytest.mark.parametrize("num_repeats", [2.5, 1.0, 3.14])
def test_action_repeat_invalid_type(num_repeats):
    """Test that non-integer num_repeats raises TypeError."""
    env = gym.make("CartPole-v1")
    with pytest.raises(TypeError, match="expected to be an integer"):
        ActionRepeat(env, num_repeats=num_repeats)


@pytest.mark.parametrize("num_repeats", [0, -1, -100])
def test_action_repeat_invalid_value(num_repeats):
    """Test that num_repeats < 1 raises ValueError."""
    env = gym.make("CartPole-v1")
    with pytest.raises(ValueError, match="equal or greater than one"):
        ActionRepeat(env, num_repeats=num_repeats)
