"""Tests that the `wrapper_checker` runs as expected and all errors are possible."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces, wrappers
from gymnasium.utils.wrapper_checker import check_wrapper
from tests.testing_env import GenericTestEnv


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


# ============= Built-in wrapper tests =============
# Test check_wrapper against every wrapper that Gymnasium ships with.
# Grouped by the type of environment they require.


def _make_cartpole():
    """Create a raw CartPole env (Discrete action, Box obs)."""
    return gym.make("CartPole-v1", disable_env_checker=True).unwrapped


def _make_continuous():
    """Create a raw MountainCarContinuous env (Box action, Box obs)."""
    return gym.make("MountainCarContinuous-v0", disable_env_checker=True).unwrapped


def _builtin_wrappers_cartpole():
    """Wrappers that work with CartPole (Discrete actions, Box obs)."""
    # Common wrappers, work with any env
    yield wrappers.TimeLimit(_make_cartpole(), max_episode_steps=100)
    yield wrappers.Autoreset(_make_cartpole())
    yield wrappers.PassiveEnvChecker(_make_cartpole())
    yield wrappers.OrderEnforcing(_make_cartpole())
    yield wrappers.RecordEpisodeStatistics(_make_cartpole())

    # Observation wrappers, work with Box obs
    yield wrappers.FlattenObservation(_make_cartpole())
    yield wrappers.DtypeObservation(_make_cartpole(), dtype=np.float64)
    yield wrappers.NormalizeObservation(_make_cartpole())
    yield wrappers.ReshapeObservation(_make_cartpole(), shape=(2, 2))
    # RescaleObservation needs finite bounds, so use MountainCar (all bounds finite)
    yield wrappers.RescaleObservation(_make_continuous(), min_obs=-1.0, max_obs=1.0)
    yield wrappers.DelayObservation(_make_cartpole(), delay=1)
    yield wrappers.FrameStackObservation(_make_cartpole(), stack_size=3)
    yield wrappers.MaxAndSkipObservation(_make_cartpole(), skip=2)
    yield wrappers.TransformObservation(
        _make_cartpole(), func=lambda obs: obs, observation_space=None
    )

    # Reward wrappers, work with any env
    yield wrappers.ClipReward(_make_cartpole(), min_reward=-1.0, max_reward=1.0)
    yield wrappers.TransformReward(_make_cartpole(), func=lambda r: r)
    yield wrappers.NormalizeReward(_make_cartpole())

    # Action wrappers, StickyAction works with any env
    yield wrappers.StickyAction(_make_cartpole(), repeat_action_probability=0.25)

    # TimeAwareObservation needs a TimeLimit wrapper
    yield wrappers.TimeAwareObservation(
        wrappers.TimeLimit(_make_cartpole(), max_episode_steps=100)
    )


def _builtin_wrappers_continuous():
    """Wrappers that require continuous (Box) action spaces."""
    yield wrappers.ClipAction(_make_continuous())
    yield wrappers.RescaleAction(_make_continuous(), min_action=-0.5, max_action=0.5)
    yield wrappers.TransformAction(
        _make_continuous(), func=lambda a: a, action_space=None
    )


@pytest.mark.parametrize(
    "wrapped_env",
    list(_builtin_wrappers_cartpole()),
    ids=lambda env: type(env).__name__,
)
def test_builtin_wrappers_cartpole(wrapped_env):
    """Check that all built-in wrappers pass check_wrapper with CartPole."""
    check_wrapper(wrapped_env, skip_render_check=True)
    wrapped_env.close()


@pytest.mark.parametrize(
    "wrapped_env",
    list(_builtin_wrappers_continuous()),
    ids=lambda env: type(env).__name__,
)
def test_builtin_wrappers_continuous(wrapped_env):
    """Check that all built-in wrappers pass check_wrapper with a continuous action env."""
    check_wrapper(wrapped_env, skip_render_check=True)
    wrapped_env.close()


# ============= Special environment wrapper tests =============
# These wrappers need custom environments (image obs, finite bounds, rendering).
# We use GenericTestEnv to create the right setup without external dependencies.


def _make_image_env():
    """Create an env with RGB image observations (25x25x3, uint8)."""
    return GenericTestEnv(
        observation_space=spaces.Box(0, 255, shape=(25, 25, 3), dtype=np.uint8)
    )


def _make_finite_obs_env():
    """Create an env with finite-bounded Box observations."""
    return GenericTestEnv(
        observation_space=spaces.Box(-1.2, 0.6, shape=(2,), dtype=np.float32)
    )


def _make_finite_action_env():
    """Create an env with finite-bounded Box actions."""
    return GenericTestEnv(
        action_space=spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    )


def _render_func(self):
    """A render function that returns a fake RGB frame."""
    return np.zeros((32, 32, 3), dtype=np.uint8)


def _make_render_env():
    """Create an env that supports rgb_array rendering."""
    return GenericTestEnv(
        render_mode="rgb_array",
        render_func=_render_func,
        metadata={"render_modes": ["rgb_array"], "render_fps": 30},
    )


def _builtin_wrappers_special():
    """Wrappers that need special environments (images, finite bounds, rendering)."""
    # GrayscaleObservation needs RGB image observations
    yield wrappers.GrayscaleObservation(_make_image_env())

    # DiscretizeObservation needs finite-bounded Box observations
    yield wrappers.DiscretizeObservation(_make_finite_obs_env(), bins=5)

    # DiscretizeAction needs finite-bounded Box actions
    yield wrappers.DiscretizeAction(_make_finite_action_env(), bins=5)

    # RenderCollection needs render_mode="rgb_array"
    yield wrappers.RenderCollection(_make_render_env())

    # AddRenderObservation needs render_mode="rgb_array", render_only mode
    yield wrappers.AddRenderObservation(_make_render_env(), render_only=True)


@pytest.mark.parametrize(
    "wrapped_env",
    list(_builtin_wrappers_special()),
    ids=lambda env: type(env).__name__,
)
def test_builtin_wrappers_special(wrapped_env):
    """Check that wrappers needing special environments pass check_wrapper."""
    check_wrapper(wrapped_env, skip_render_check=True)
    wrapped_env.close()
