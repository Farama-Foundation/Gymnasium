"""Test suite for ClipReward wrapper."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.error import InvalidBound
from gymnasium.wrappers import ClipReward
from tests.wrappers.utils import DISCRETE_ACTION, ENV_ID, SEED


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected_reward"),
    [(None, 0.5, 0.5), (0, None, 1), (0, 0.5, 0.5)],
)
def test_clip_reward_wrapper(lower_bound, upper_bound, expected_reward):
    """Test reward clipping.

    Test if reward is correctly clipped accordingly to the input args.
    """
    env = gym.make(ENV_ID)
    env = ClipReward(env, lower_bound, upper_bound)
    env.reset(seed=SEED)
    _, rew, _, _, _ = env.step(DISCRETE_ACTION)

    assert rew == expected_reward


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound"),
    [(None, None), (1, -1), (np.array([1, 1]), np.array([0, 0]))],
)
def test_clip_reward_incorrect_params(lower_bound, upper_bound):
    """Test reward clipping with incorrect params.

    Test whether passing wrong params to clip_rewards correctly raise an exception.
    clip_rewards should raise an exception if, both low and upper bound of reward are `None`
    or if upper bound is lower than lower bound.
    """
    env = gym.make(ENV_ID)

    with pytest.raises(InvalidBound):
        ClipReward(env, lower_bound, upper_bound)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("as_array", [False, True])
def test_clip_reward_signed_integer_bounds(dtype, as_array):
    """Accept ordered integer bounds even when their difference would overflow."""
    limits = np.iinfo(dtype)
    lower_bound, upper_bound = dtype(limits.min), dtype(limits.max)
    if as_array:
        lower_bound, upper_bound = np.array([lower_bound]), np.array([upper_bound])

    with gym.make(ENV_ID) as env:
        with np.errstate(over="raise"):
            wrapped = ClipReward(env, lower_bound, upper_bound)
        wrapped.reset(seed=SEED)
        _, reward, _, _, _ = wrapped.step(DISCRETE_ACTION)
        np.testing.assert_array_equal(reward, np.clip(1.0, lower_bound, upper_bound))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("as_array", [False, True])
def test_clip_reward_reversed_unsigned_bounds(dtype, as_array):
    """Reject reversed unsigned bounds instead of wrapping their difference."""
    lower_bound, upper_bound = dtype(2), dtype(1)
    if as_array:
        lower_bound, upper_bound = np.array([lower_bound]), np.array([upper_bound])

    with gym.make(ENV_ID) as env:
        with np.errstate(over="raise"), pytest.raises(InvalidBound):
            ClipReward(env, lower_bound, upper_bound)
