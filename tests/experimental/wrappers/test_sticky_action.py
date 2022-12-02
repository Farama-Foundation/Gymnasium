"""Test suite for StickyActionV0."""
import pytest

from gymnasium.error import InvalidProbability
from gymnasium.experimental.wrappers import StickyActionV0
from tests.testing_env import GenericTestEnv

SEED = 42

DELAY = 3
NUM_STEPS = 10


def step_fn(self, action):
    return action


def test_sticky_action():
    env = StickyActionV0(GenericTestEnv(step_fn=step_fn), repeat_action_probability=0.5)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    previous_action = None
    for _ in range(NUM_STEPS):
        input_action = env.action_space.sample()
        executed_action = env.step(input_action)

        if executed_action != input_action:
            assert executed_action == previous_action
        else:
            assert executed_action == input_action

        previous_action = input_action


@pytest.mark.parametrize(("repeat_action_probability"), [-1, 1, 1.5])
def test_sticky_action_raise(repeat_action_probability):
    with pytest.raises(InvalidProbability):
        StickyActionV0(
            GenericTestEnv(), repeat_action_probability=repeat_action_probability
        )
