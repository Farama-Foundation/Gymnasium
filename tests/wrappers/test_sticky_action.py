"""Test suite for StickyAction wrapper."""

import pytest

from gymnasium.error import InvalidBound, InvalidProbability
from gymnasium.spaces import Discrete
from gymnasium.wrappers import StickyAction
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import record_action_as_obs_step


@pytest.mark.parametrize(
    "repeat_action_probability,repeat_action_duration,actions,expected_action",
    [
        (0.25, 1, [0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 2, 3, 3, 3, 6, 6]),
        (0.25, 2, [0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 3, 4, 4, 4, 4]),
        (0.25, (1, 3), [0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 4, 4, 4, 4]),
    ],
)
def test_sticky_action(
    repeat_action_probability, repeat_action_duration, actions, expected_action
):
    """Tests the sticky action wrapper."""
    env = StickyAction(
        GenericTestEnv(
            step_func=record_action_as_obs_step, observation_space=Discrete(7)
        ),
        repeat_action_probability=repeat_action_probability,
        repeat_action_duration=repeat_action_duration,
    )
    env.reset(seed=11)

    assert len(actions) == len(expected_action)
    for action, action_taken in zip(actions, expected_action):
        executed_action, _, _, _, _ = env.step(action)
        assert executed_action == action_taken


@pytest.mark.parametrize("repeat_action_probability", [-1, 1, 1.5])
def test_sticky_action_raise_probability(repeat_action_probability):
    """Tests the stick action wrapper with probabilities that should raise an error."""
    with pytest.raises(InvalidProbability):
        StickyAction(
            GenericTestEnv(), repeat_action_probability=repeat_action_probability
        )


@pytest.mark.parametrize(
    "repeat_action_duration",
    [
        -4,
        0,
        (0, 0),
        (4, 2),
        [1, 2],
    ],
)
def test_sticky_action_raise_duration(repeat_action_duration):
    """Tests the stick action wrapper with durations that should raise an error."""
    with pytest.raises((ValueError, InvalidBound)):
        StickyAction(
            GenericTestEnv(), 0.5, repeat_action_duration=repeat_action_duration
        )
