"""Test suite for LambdaActionV0."""
import numpy as np
import pytest

import gymnasium as gym
from gymnasium.experimental.wrappers import ClipActionV0

SEED = 42


@pytest.mark.parametrize(
    ("env", "action_unclipped_env", "action_clipped_env"),
    (
        [
            # MountainCar action space: Box(-1.0, 1.0, (1,), float32)
            gym.make("MountainCarContinuous-v0"),
            np.array([1]),
            np.array([1.5]),
        ],
        [
            # BipedalWalker action space: Box(-1.0, 1.0, (4,), float32)
            gym.make("BipedalWalker-v3"),
            np.array([1, 1, 1, 1]),
            np.array([10, 10, 10, 10]),
        ],
        [
            # BipedalWalker action space: Box(-1.0, 1.0, (4,), float32)
            gym.make("BipedalWalker-v3"),
            np.array([0.5, 0.5, 1, 1]),
            np.array([0.5, 0.5, 10, 10]),
        ],
    ),
)
def test_clip_actions_v0(env, action_unclipped_env, action_clipped_env):
    """Tests if actions out of bound are correctly clipped.

    Tests whether out of bound actions for the wrapped
    environments are correctly clipped.
    """
    env.reset(seed=SEED)
    obs, _, _, _, _ = env.step(action_unclipped_env)

    env.reset(seed=SEED)
    wrapped_env = ClipActionV0(env)
    wrapped_obs, _, _, _, _ = wrapped_env.step(action_clipped_env)

    assert np.alltrue(obs == wrapped_obs)
