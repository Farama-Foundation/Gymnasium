"""Tests that apply to every vector action wrapper, e.g., that the caller's actions are not modified in-place."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AutoresetMode, SyncVectorEnv, VectorActionWrapper
from tests.testing_env import GenericTestEnv

BOX_ACTION_SPACE = Box(-1, 1, shape=(3,), dtype=np.float32)
DICT_ACTION_SPACE = Dict({"a": BOX_ACTION_SPACE, "b": Box(0, 1, shape=(2, 2))})

# The vector action wrappers with the constructor kwargs and sub-environment action space to test them with
VECTOR_ACTION_WRAPPERS: dict[str, list[tuple[dict[str, Any], gym.Space]]] = {
    "TransformAction": [
        ({"func": lambda actions: 0.5 * actions}, BOX_ACTION_SPACE),
        (
            {"func": lambda actions: {key: -value for key, value in actions.items()}},
            DICT_ACTION_SPACE,
        ),
    ],
    "VectorizeTransformAction": [
        # The same action space, previously, the transformed actions were written into the caller's actions
        (
            {
                "wrapper": gym.wrappers.TransformAction,
                "func": lambda action: 0.5 * action,
                "action_space": None,
            },
            BOX_ACTION_SPACE,
        ),
        (
            {
                "wrapper": gym.wrappers.TransformAction,
                "func": lambda action: {key: -value for key, value in action.items()},
                "action_space": None,
            },
            DICT_ACTION_SPACE,
        ),
        # A different action space
        (
            {
                "wrapper": gym.wrappers.TransformAction,
                "func": lambda action: np.tanh(action),
                "action_space": Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            },
            BOX_ACTION_SPACE,
        ),
    ],
    "ClipAction": [({}, BOX_ACTION_SPACE)],
    "RescaleAction": [({"min_action": 0.0, "max_action": 2.0}, BOX_ACTION_SPACE)],
}


def _is_vector_action_wrapper(name: str) -> bool:
    wrapper = getattr(gym.wrappers.vector, name)
    return inspect.isclass(wrapper) and issubclass(wrapper, VectorActionWrapper)


def test_all_vector_action_wrappers_are_tested():
    """Checks that every vector action wrapper is included in `VECTOR_ACTION_WRAPPERS`."""
    vector_action_wrappers = {
        name for name in gym.wrappers.vector.__all__ if _is_vector_action_wrapper(name)
    }
    assert vector_action_wrappers == set(VECTOR_ACTION_WRAPPERS)


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
@pytest.mark.parametrize(
    "wrapper_name, kwargs, env_action_space",
    [
        (wrapper_name, kwargs, env_action_space)
        for wrapper_name, cases in VECTOR_ACTION_WRAPPERS.items()
        for kwargs, env_action_space in cases
    ],
)
def test_actions_not_modified_in_place(
    wrapper_name: str,
    kwargs: dict[str, Any],
    env_action_space: gym.Space,
    autoreset_mode: AutoresetMode,
    num_envs: int = 3,
    num_steps: int = 20,
):
    """Checks that the vector action wrapper doesn't modify the caller's actions in-place."""
    envs = getattr(gym.wrappers.vector, wrapper_name)(
        SyncVectorEnv(
            [
                lambda: GenericTestEnv(action_space=env_action_space)
                for _ in range(num_envs)
            ],
            autoreset_mode=autoreset_mode,
        ),
        **kwargs,
    )
    envs.reset(seed=123)
    envs.action_space.seed(123)

    for _ in range(num_steps):
        actions = envs.action_space.sample()
        actions_copy = deepcopy(actions)
        envs.step(actions)
        assert data_equivalence(actions, actions_copy), (
            f"{wrapper_name} modified the actions in-place"
        )

    envs.close()
