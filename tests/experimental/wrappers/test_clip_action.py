"""Test suite for ClipActionV0."""

import numpy as np

from gymnasium.experimental.wrappers import ClipActionV0
from gymnasium.spaces import Box
from tests.experimental.wrappers.utils import record_action_step
from tests.testing_env import GenericTestEnv


def test_clip_action_wrapper():
    """Test that the action is correctly clipped to the base environment action space."""
    env = GenericTestEnv(
        action_space=Box(np.array([0, 0, 3]), np.array([1, 2, 4])),
        step_func=record_action_step,
    )
    wrapped_env = ClipActionV0(env)

    sampled_action = np.array([-1, 5, 3.5], dtype=np.float32)
    assert sampled_action not in env.action_space
    assert sampled_action in wrapped_env.action_space

    _, _, _, _, info = wrapped_env.step(sampled_action)
    assert np.all(info["action"] in env.action_space)
    assert np.all(info["action"] == np.array([0, 2, 3.5]))
