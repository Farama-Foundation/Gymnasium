"""Test lambda reward wrapper."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.experimental.wrappers import LambdaRewardV0
from tests.experimental.wrappers.utils import DISCRETE_ACTION, ENV_ID, NUM_ENVS, SEED


@pytest.mark.parametrize(
    ("reward_fn", "expected_reward"),
    [(lambda r: 2 * r + 1, 3)],
)
def test_lambda_reward(reward_fn, expected_reward):
    """Test lambda reward.

    Tests if function is correctly applied
    to reward.
    """
    env = gym.make(ENV_ID)
    env = LambdaRewardV0(env, reward_fn)
    env.reset(seed=SEED)

    _, rew, _, _, _ = env.step(DISCRETE_ACTION)

    assert rew == expected_reward


@pytest.mark.parametrize(
    (
        "reward_fn",
        "expected_reward",
    ),
    [(lambda r: 2 * r + 1, 3)],
)
def test_lambda_reward_within_vector(reward_fn, expected_reward):
    """Test lambda reward in vectorized environment.

    Tests if function is correctly applied
    to reward in a vectorized environment.
    """
    actions = [DISCRETE_ACTION for _ in range(NUM_ENVS)]
    env = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
    env = LambdaRewardV0(env, reward_fn)
    env.reset(seed=SEED)

    _, rew, _, _, _ = env.step(actions)

    assert np.alltrue(rew == expected_reward)
