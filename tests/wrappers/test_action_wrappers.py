"""Tests that apply to every action wrapper, e.g., that the caller's action is not modified in-place."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.utils.env_checker import data_equivalence
from tests.testing_env import GenericTestEnv

BOX_ACTION_SPACE = Box(-1, 1, shape=(3,), dtype=np.float32)

# The action wrappers with the constructor kwargs and base environment action space to test them with
ACTION_WRAPPERS: dict[str, list[tuple[dict[str, Any], gym.Space]]] = {
    "TransformAction": [
        (
            {"func": lambda action: 0.5 * action, "action_space": None},
            BOX_ACTION_SPACE,
        ),
        (
            {
                "func": lambda action: {key: -value for key, value in action.items()},
                "action_space": None,
            },
            Dict({"a": BOX_ACTION_SPACE, "b": Box(0, 1, shape=(2, 2))}),
        ),
    ],
    "ClipAction": [({}, BOX_ACTION_SPACE)],
    "RescaleAction": [({"min_action": 0.0, "max_action": 2.0}, BOX_ACTION_SPACE)],
    "DiscretizeAction": [
        ({"bins": 5, "multidiscrete": True}, BOX_ACTION_SPACE),
        ({"bins": 5, "multidiscrete": False}, BOX_ACTION_SPACE),
    ],
    "StickyAction": [({"repeat_action_probability": 0.5}, BOX_ACTION_SPACE)],
    "RepeatAction": [({"num_repeats": 3}, BOX_ACTION_SPACE)],
}


def _is_action_wrapper(name: str) -> bool:
    wrapper = getattr(gym.wrappers, name)
    return inspect.isclass(wrapper) and (
        issubclass(wrapper, gym.ActionWrapper)
        or (issubclass(wrapper, gym.Wrapper) and name.endswith("Action"))
    )


def test_all_action_wrappers_are_tested():
    """Checks that every action wrapper is included in `ACTION_WRAPPERS`."""
    action_wrappers = {
        name for name in gym.wrappers.__all__ if _is_action_wrapper(name)
    }
    assert action_wrappers == set(ACTION_WRAPPERS)


@pytest.mark.parametrize(
    "wrapper_name, kwargs, env_action_space",
    [
        (wrapper_name, kwargs, env_action_space)
        for wrapper_name, cases in ACTION_WRAPPERS.items()
        for kwargs, env_action_space in cases
    ],
)
def test_action_not_modified_in_place(
    wrapper_name: str,
    kwargs: dict[str, Any],
    env_action_space: gym.Space,
    num_steps: int = 20,
):
    """Checks that the action wrapper doesn't modify the caller's action in-place."""
    env = getattr(gym.wrappers, wrapper_name)(
        GenericTestEnv(action_space=env_action_space), **kwargs
    )
    env.reset(seed=123)
    env.action_space.seed(123)

    for _ in range(num_steps):
        action = env.action_space.sample()
        action_copy = deepcopy(action)
        env.step(action)
        assert data_equivalence(action, action_copy), (
            f"{wrapper_name} modified the action in-place"
        )

    env.close()
