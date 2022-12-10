"""Test suite for ClipRewardV0."""
import numpy as np
import pytest

import gymnasium as gym
from gymnasium.error import InvalidBound
from gymnasium.experimental.wrappers import ClipRewardV0
from tests.envs.test_envs import SEED
from tests.experimental.wrappers.test_lambda_rewards import (
    DISCRETE_ACTION,
    ENV_ID,
    NUM_ENVS,
)


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected_reward"),
    [(None, 0.5, 0.5), (0, None, 1), (0, 0.5, 0.5)],
)
def test_clip_reward(lower_bound, upper_bound, expected_reward):
    """Test reward clipping.

    Test if reward is correctly clipped accordingly to the input args.
    """
    env = gym.make(ENV_ID)
    env = ClipRewardV0(env, lower_bound, upper_bound)
    env.reset(seed=SEED)
    _, rew, _, _, _ = env.step(DISCRETE_ACTION)

    assert rew == expected_reward


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected_reward"),
    [(None, 0.5, 0.5), (0, None, 1), (0, 0.5, 0.5)],
)
def test_clip_reward_within_vector(lower_bound, upper_bound, expected_reward):
    """Test reward clipping in vectorized environment.

    Test if reward is correctly clipped accordingly to the input args in a vectorized environment.
    """
    actions = [DISCRETE_ACTION for _ in range(NUM_ENVS)]

    env = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
    env = ClipRewardV0(env, lower_bound, upper_bound)
    env.reset(seed=SEED)

    _, rew, _, _, _ = env.step(actions)

    assert np.alltrue(rew == expected_reward)


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
        ClipRewardV0(env, lower_bound, upper_bound)
