"""Test suite for RescaleActionV0."""
import numpy as np
import pytest

import gymnasium
from gymnasium.wrappers import RescaleActionV0

SEED = 42
NUM_ENVS = 3


@pytest.mark.parametrize(
    ("env", "low", "high", "action", "scaled_action"),
    [
        (
            # BipedalWalker action space: Box(-1.0, 1.0, (4,), float32)
            gymnasium.make("BipedalWalker-v3"),
            -0.5,
            0.5,
            np.array([1, 1, 1, 1]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ),
    ],
)
def test_rescale_actions_v0_box(env, low, high, action, scaled_action):
    """Test action rescaling."""
    env.reset(seed=SEED)
    obs, _, _, _, _ = env.step(action)

    env.reset(seed=SEED)
    wrapped_env = RescaleActionV0(env, low, high)

    obs_scaled, _, _, _, _ = wrapped_env.step(scaled_action)

    assert np.alltrue(obs == obs_scaled)
