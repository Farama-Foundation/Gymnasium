"""Test suite for StickyAction wrapper."""

from copy import deepcopy

import numpy as np
import pytest

from gymnasium.error import InvalidBound, InvalidProbability
from gymnasium.spaces import Box, Dict, Discrete, Tuple
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.wrappers import StickyAction
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import record_action_as_obs_step


def record_copied_action_step(self, action):
    """Return a fresh observation showing the action executed by the environment."""
    assert action in self.action_space
    return deepcopy(action), 0.0, False, False, {}


@pytest.mark.parametrize("repeat_action_duration", [1, 3, (2, 4)])
@pytest.mark.parametrize("nested", [False, True], ids=["box", "nested-dict-tuple"])
def test_sticky_action_preserves_reused_action(nested, repeat_action_duration):
    """Reusing a caller-owned action buffer must not change the previous action."""
    box = Box(-1.0, 1.0, (2,), dtype=np.float32)
    action = np.array([0.25, -0.5], dtype=np.float32)
    if nested:
        action_space = Dict(motor=box, gripper=Tuple((box, Discrete(2))))
        action = {"motor": action, "gripper": (action.copy(), 1)}
    else:
        action_space = box

    env = StickyAction(
        GenericTestEnv(
            action_space=action_space,
            observation_space=action_space,
            step_func=record_copied_action_step,
        ),
        repeat_action_probability=0.99,
        repeat_action_duration=repeat_action_duration,
    )
    env.reset(seed=0)
    previous_action = deepcopy(action)
    executed_action, *_ = env.step(action)
    assert data_equivalence(executed_action, previous_action)

    # Update the same buffer in place, including the leaves of a nested action.
    if nested:
        action["motor"][:] = 0.9
        action["gripper"][0][:] = -0.9
        action["gripper"] = (action["gripper"][0], 0)
    else:
        action[:] = 0.9
    assert action in action_space
    assert not data_equivalence(action, previous_action)

    # Seed 0 triggers a sticky series on the second step. Every repeat must
    # retain the previously executed values, rather than the updated proposal.
    duration_min, duration_max = (
        (repeat_action_duration, repeat_action_duration)
        if isinstance(repeat_action_duration, int)
        else repeat_action_duration
    )
    for _repeat in range(1, duration_max + 1):
        executed_action, *_ = env.step(action)
        assert data_equivalence(executed_action, previous_action)
        if not env.is_sticky_actions:
            break
    assert not env.is_sticky_actions
    assert duration_min <= _repeat <= duration_max

    # A reset discards the saved action, so the new episode starts with the
    # latest proposed values even when its caller keeps using the same buffer.
    env.reset(seed=0)
    executed_action, *_ = env.step(action)
    assert data_equivalence(executed_action, action)
    env.close()


def test_sticky_action_zero_probability_honors_reused_action():
    """Without sticky repeats, every in-place update is a new executed control."""
    box = Box(-1.0, 1.0, (2,), dtype=np.float32)
    env = StickyAction(
        GenericTestEnv(
            action_space=box,
            observation_space=box,
            step_func=record_copied_action_step,
        ),
        repeat_action_probability=0.0,
        repeat_action_duration=3,
    )
    env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    for value in [0.25, -0.5, 0.9]:
        action[:] = value
        executed_action, *_ = env.step(action)
        assert data_equivalence(executed_action, action)
        assert not env.is_sticky_actions
    env.close()


def test_sticky_action_reset_during_repeats():
    """Reset must discard a copied previous action and an unfinished sticky series."""
    box = Box(-1.0, 1.0, (2,), dtype=np.float32)
    env = StickyAction(
        GenericTestEnv(
            action_space=box,
            observation_space=box,
            step_func=record_copied_action_step,
        ),
        repeat_action_probability=0.99,
        repeat_action_duration=3,
    )
    env.reset(seed=0)
    action = np.array([0.25, -0.5], dtype=np.float32)
    env.step(action)
    action[:] = 0.9
    executed_action, *_ = env.step(action)
    np.testing.assert_array_equal(executed_action, [0.25, -0.5])
    assert env.is_sticky_actions

    env.reset(seed=0)
    assert env.last_action is None
    assert not env.is_sticky_actions
    assert env.num_repeats == env.repeats_taken == 0
    executed_action, *_ = env.step(action)
    assert data_equivalence(executed_action, action)
    env.close()


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
    for action, action_taken in zip(actions, expected_action, strict=True):
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
