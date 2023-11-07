"""Test suite for RescaleAction wrapper."""
import numpy as np

from gymnasium.spaces import Box
from gymnasium.wrappers import RescaleAction
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import record_action_step


def test_rescale_action_wrapper():
    """Test that the action is rescale within a min / max bound."""
    env = GenericTestEnv(
        step_func=record_action_step,
        action_space=Box(np.array([0, 1]), np.array([1, 3])),
    )
    wrapped_env = RescaleAction(
        env, min_action=np.array([-5, 0]), max_action=np.array([5, 1])
    )
    assert wrapped_env.action_space == Box(np.array([-5, 0]), np.array([5, 1]))

    for sample_action, expected_action in (
        (
            np.array([0.0, 0.5], dtype=np.float32),
            np.array([0.5, 2.0], dtype=np.float32),
        ),
        (
            np.array([-5.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ),
        (
            np.array([5.0, 1.0], dtype=np.float32),
            np.array([1.0, 3.0], dtype=np.float32),
        ),
    ):
        assert sample_action in wrapped_env.action_space

        _, _, _, _, info = wrapped_env.step(sample_action)
        assert np.all(info["action"] == expected_action)
