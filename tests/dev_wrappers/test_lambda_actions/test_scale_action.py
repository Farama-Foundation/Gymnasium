"""Test suite for ScaleActionsV0."""
import numpy as np
import pytest

import gymnasium
from gymnasium.wrappers import RescaleActionsV0

SEED = 42
NUM_ENVS = 3


@pytest.mark.parametrize(
    ("env", "args", "action", "scaled_action"),
    [
        (
            # BipedalWalker action space: Box(-1.0, 1.0, (4,), float32)
            gymnasium.make("BipedalWalker-v3"),
            (-0.5, 0.5),
            np.array([1, 1, 1, 1]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ),
    ],
)
def test_scale_actions_v0_box(env, args, action, scaled_action):
    """Test action rescaling.
    Scale action wrapper allow to rescale action
    to a new range.
    Supposed the old action space is
    `Box(-1, 1, (1,))` and we rescale to
    `Box(-0.5, 0.5, (1,))`, an action  with value
    `0.5` will have the same effect of an action with value
    `1.0` on the unwrapped env.
    """
    env.reset(seed=SEED)
    obs, _, _, _, _ = env.step(action)

    env.reset(seed=SEED)
    wrapped_env = RescaleActionsV0(env, args)

    obs_scaled, _, _, _, _ = wrapped_env.step(scaled_action)

    assert np.alltrue(obs == obs_scaled)
