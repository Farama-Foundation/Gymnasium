"""Test suite for LambdaAction wrapper."""

from gymnasium.spaces import Box
from gymnasium.wrappers import TransformAction
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import record_action_step


def test_lambda_action_wrapper():
    """Tests LambdaAction through checking that the action taken is transformed by function."""
    env = GenericTestEnv(step_func=record_action_step)
    wrapped_env = TransformAction(env, lambda action: action - 2, Box(2, 3))

    sampled_action = wrapped_env.action_space.sample()
    assert sampled_action not in env.action_space

    _, _, _, _, info = wrapped_env.step(sampled_action)
    assert info["action"] in env.action_space
    assert sampled_action - 2 == info["action"]
