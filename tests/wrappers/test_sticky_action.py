"""Test suite for StickyAction wrapper."""
import numpy as np
import pytest

from gymnasium.error import InvalidProbability
from gymnasium.wrappers import StickyAction
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import NUM_STEPS, record_action_as_obs_step


def test_sticky_action():
    """Tests the sticky action wrapper."""
    env = StickyAction(
        GenericTestEnv(step_func=record_action_as_obs_step),
        repeat_action_probability=0.5,
    )

    previous_action = None
    for _ in range(NUM_STEPS):
        input_action = env.action_space.sample()
        executed_action, _, _, _, _ = env.step(input_action)

        assert np.all(executed_action == input_action) or np.all(
            executed_action == previous_action
        )
        previous_action = executed_action


@pytest.mark.parametrize("repeat_action_probability", [-1, 1, 1.5])
def test_sticky_action_raise(repeat_action_probability):
    """Tests the stick action wrapper with probabilities that should raise an error."""
    with pytest.raises(InvalidProbability):
        StickyAction(
            GenericTestEnv(), repeat_action_probability=repeat_action_probability
        )
