"""Test suit for lambda action wrappers: LambdaAction, ClipAction, RescaleAction."""
import numpy as np

from gymnasium.experimental.wrappers import (
    ClipActionV0,
    LambdaActionV0,
    RescaleActionV0,
)
from gymnasium.spaces import Box
from tests.testing_env import GenericTestEnv


SEED = 42


def _record_action_step_func(self, action):
    return 0, 0, False, False, {"action": action}


def test_lambda_action_wrapper():
    """Tests LambdaAction through checking that the action taken is transformed by function."""
    env = GenericTestEnv(step_func=_record_action_step_func)
    wrapped_env = LambdaActionV0(env, lambda action: action - 2, Box(2, 3))

    sampled_action = wrapped_env.action_space.sample()
    assert sampled_action not in env.action_space

    _, _, _, _, info = wrapped_env.step(sampled_action)
    assert info["action"] in env.action_space
    assert sampled_action - 2 == info["action"]


def test_clip_action_wrapper():
    """Test that the action is correctly clipped to the base environment action space."""
    env = GenericTestEnv(
        action_space=Box(np.array([0, 0, 3]), np.array([1, 2, 4])),
        step_func=_record_action_step_func,
    )
    wrapped_env = ClipActionV0(env)

    sampled_action = np.array([-1, 5, 3.5], dtype=np.float32)
    assert sampled_action not in env.action_space
    assert sampled_action in wrapped_env.action_space

    _, _, _, _, info = wrapped_env.step(sampled_action)
    assert np.all(info["action"] in env.action_space)
    assert np.all(info["action"] == np.array([0, 2, 3.5]))


def test_rescale_action_wrapper():
    """Test that the action is rescale within a min / max bound."""
    env = GenericTestEnv(
        step_func=_record_action_step_func,
        action_space=Box(np.array([0, 1]), np.array([1, 3])),
    )
    wrapped_env = RescaleActionV0(
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
