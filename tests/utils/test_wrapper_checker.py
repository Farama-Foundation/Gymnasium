"""Tests that the `wrapper_checker` runs as expected and all errors are possible."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.wrapper_checker import check_wrapper


def test_correct_wrapper_passes():
    """A correct wrapper should pass all checks without errors."""
    env = gym.make("CartPole-v1")
    check_wrapper(env)


def test_correct_wrapper_mountaincar():
    """Test with a different environment to make sure it's not CartPole-specific."""
    env = gym.make("MountainCar-v0")
    check_wrapper(env)


def test_not_a_wrapper():
    """Passing a non-wrapper object should raise TypeError."""
    with pytest.raises(TypeError, match="must inherit"):
        check_wrapper("not a wrapper")


def test_raw_env_not_a_wrapper():
    """Passing an unwrapped environment should raise TypeError."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    with pytest.raises(TypeError, match="must inherit"):
        check_wrapper(env)


class BadObsWrapper(gym.ObservationWrapper):
    """A wrapper that lies about its observation space."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, shape=(10, 10))

    def observation(self, obs):
        return obs  # BUG: returns original obs, doesn't match 10x10 space


def test_bad_observation_wrapper():
    """A wrapper whose observations don't match the declared space should fail."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    wrapped = BadObsWrapper(env)
    with pytest.raises(AssertionError, match="not in observation_space"):
        check_wrapper(wrapped)


class BadStepWrapper(gym.Wrapper):
    """A wrapper that returns the wrong number of values from step()."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated  # BUG: only 3 values instead of 5


def test_bad_step_return():
    """A wrapper whose step() returns wrong number of values should fail."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    wrapped = BadStepWrapper(env)
    with pytest.raises(AssertionError, match="must return 5 values"):
        check_wrapper(wrapped)


class BadResetWrapper(gym.Wrapper):
    """A wrapper that returns only the observation from reset(), missing info."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs  # BUG: should return (obs, info) tuple


def test_bad_reset_return():
    """A wrapper whose reset() doesn't return a tuple should fail."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    wrapped = BadResetWrapper(env)
    with pytest.raises(AssertionError, match="must return a tuple"):
        check_wrapper(wrapped)


# ============= Real RL usage tests =============
# These tests create custom wrappers, run an actual RL loop through them,
# and then verify them with check_wrapper().


class DoubleRewardWrapper(gym.RewardWrapper):
    """A wrapper that doubles the reward from every step."""

    def reward(self, reward):
        return reward * 2.0


def test_reward_wrapper_rl_loop():
    """Test a RewardWrapper with a real RL loop, then verify with check_wrapper."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    wrapped = DoubleRewardWrapper(env)

    # Run an RL loop — the agent interacts with the wrapper, not the raw env
    obs, info = wrapped.reset(seed=42)
    for _ in range(5):
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)

        # CartPole gives reward=1.0 per step, so doubled = 2.0
        assert reward == 2.0, f"Expected doubled reward 2.0, got {reward}"

        if terminated or truncated:
            obs, info = wrapped.reset()

    wrapped.close()

    # Now verify the wrapper passes all structural checks
    fresh_env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    check_wrapper(DoubleRewardWrapper(fresh_env))


class NegateObsWrapper(gym.ObservationWrapper):
    """A wrapper that negates all observation values."""

    def observation(self, obs):
        return -obs


def test_observation_wrapper_rl_loop():
    """Test an ObservationWrapper with a real RL loop, then verify with check_wrapper."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped

    # Get a baseline observation without the wrapper
    raw_obs, _ = env.reset(seed=42)

    # Now wrap and get the same observation
    env2 = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    wrapped = NegateObsWrapper(env2)
    wrapped_obs, _ = wrapped.reset(seed=42)

    # The wrapped observation should be the negation of the raw one
    np.testing.assert_array_almost_equal(wrapped_obs, -raw_obs)

    # Run a few steps and verify observations stay negated
    for _ in range(3):
        action = wrapped.action_space.sample()
        wrapped_obs, reward, terminated, truncated, info = wrapped.step(action)

        # All observation values should be negated (most will be non-zero)
        # We can't compare with raw env here since actions are random,
        # but we CAN check the obs is in the space
        assert wrapped_obs in wrapped.observation_space

        if terminated or truncated:
            break

    wrapped.close()

    # Verify with check_wrapper
    fresh_env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    check_wrapper(NegateObsWrapper(fresh_env))


class FlipActionWrapper(gym.ActionWrapper):
    """A wrapper that flips CartPole actions: left becomes right, right becomes left."""

    def action(self, action):
        return 1 - action  # 0→1, 1→0


def test_action_wrapper_rl_loop():
    """Test an ActionWrapper with a real RL loop, then verify with check_wrapper."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    wrapped = FlipActionWrapper(env)

    # Run an RL loop through the wrapper
    obs, info = wrapped.reset(seed=42)
    for _ in range(5):
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)

        # The environment should still work — just with flipped controls
        assert obs in wrapped.observation_space
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        if terminated or truncated:
            obs, info = wrapped.reset()

    wrapped.close()

    # Verify with check_wrapper
    fresh_env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    check_wrapper(FlipActionWrapper(fresh_env))


# ============= RL failure case tests =============
# These test wrappers that LOOK like they work but produce bad data
# during an actual RL loop. check_wrapper() should be able to catch them.


class StringRewardWrapper(gym.RewardWrapper):
    """A wrapper that accidentally returns a string reward instead of a number."""

    def reward(self, reward):
        return "good job"  # BUG: reward should be a number


def test_string_reward_wrapper():
    """A wrapper returning a non-numeric reward should trigger a warning."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    wrapped = StringRewardWrapper(env)

    # check_wrapper uses logger.warn for bad rewards (not assert),
    # so we catch the warning
    with pytest.warns(UserWarning, match="must be a float"):
        check_wrapper(wrapped)


class WrongShapeObsWrapper(gym.ObservationWrapper):
    """A wrapper that returns a single number instead of the full observation array."""

    def __init__(self, env):
        super().__init__(env)
        # Declares a Box space expecting a single float
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def observation(self, obs):
        return obs  # BUG: returns shape (4,) but space expects shape (1,)


def test_wrong_shape_obs_wrapper():
    """A wrapper whose observations don't match the declared space shape should fail."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    wrapped = WrongShapeObsWrapper(env)
    with pytest.raises(AssertionError, match="not in observation_space"):
        check_wrapper(wrapped)


class IntTerminatedWrapper(gym.Wrapper):
    """A wrapper that returns an int for terminated instead of a bool."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, 1, truncated, info  # BUG: 1 instead of True


def test_int_terminated_wrapper():
    """A wrapper returning int instead of bool for terminated should fail."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    wrapped = IntTerminatedWrapper(env)
    with pytest.raises(AssertionError, match="terminated from step.*must be a bool"):
        check_wrapper(wrapped)


class NoneInfoWrapper(gym.Wrapper):
    """A wrapper that returns None for info instead of a dict."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, None  # BUG: None instead of {}


def test_none_info_wrapper():
    """A wrapper returning None instead of dict for info should fail."""
    env = gym.make("CartPole-v1", disable_env_checker=True).unwrapped
    wrapped = NoneInfoWrapper(env)
    with pytest.raises(AssertionError, match="info from step.*must be a dict"):
        check_wrapper(wrapped)
