"""Test suite for LambdaReward wrapper."""

from gymnasium.wrappers import TransformReward
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import record_action_as_record_step


def test_lambda_reward():
    env = GenericTestEnv(step_func=record_action_as_record_step)
    wrapped_env = TransformReward(env, lambda r: 2 * r + 1)

    _, rew, _, _, _ = wrapped_env.step(0)
    assert rew == 1
    _, rew, _, _, _ = wrapped_env.step(1)
    assert rew == 3
