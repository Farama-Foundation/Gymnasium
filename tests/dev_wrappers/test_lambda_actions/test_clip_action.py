"""Test suite for LambdaActionV0."""
from collections import OrderedDict

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import ClipActionV0
from tests.dev_wrappers.mock_data import (
    DICT_SPACE,
    DICT_WITHIN_TUPLE_SPACE,
    DOUBLY_NESTED_DICT_SPACE,
    DOUBLY_NESTED_TUPLE_SPACE,
    NESTED_DICT_SPACE,
    NESTED_TUPLE_SPACE,
    NEW_BOX_HIGH,
    NEW_BOX_LOW,
    TUPLE_SPACE,
    TUPLE_WITHIN_DICT_SPACE,
)
from tests.testing_env import GenericTestEnv

NUM_ENVS = 3
SEED = 42


def env_step_fn(self, action):
    return 0, 0, False, False, {"action": action}


@pytest.mark.parametrize(
    ("env", "args", "action_unclipped_env", "action_clipped_env"),
    (
        [
            # MountainCar action space: Box(-1.0, 1.0, (1,), float32)
            gym.make("MountainCarContinuous-v0"),
            (np.array([-0.5], dtype="float32"), np.array([0.5], dtype="float32")),
            np.array([0.5]),
            np.array([1]),
        ],
        [
            # BipedalWalker action space: Box(-1.0, 1.0, (4,), float32)
            gym.make("BipedalWalker-v3"),
            (-0.5, 0.5),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([10, 10, 10, 10]),
        ],
        [
            # BipedalWalker action space: Box(-1.0, 1.0, (4,), float32)
            gym.make("BipedalWalker-v3"),
            (
                np.array([-0.5, -1, -1, -1], dtype="float32"),
                np.array([0.5, 0.5, 1, 1], dtype="float32"),
            ),
            np.array([0.5, 0.5, 1, 1]),
            np.array([10, 10, 10, 10]),
        ],
    ),
)
def test_clip_actions_v0(env, args, action_unclipped_env, action_clipped_env):
    """Tests if actions out of bound are correctly clipped.

    Tests whether out of bound actions for the wrapped
    environments are correctly clipped.
    """
    env.reset(seed=SEED)
    obs, _, _, _, _ = env.step(action_unclipped_env)

    env.reset(seed=SEED)
    wrapped_env = ClipActionV0(env, args)
    wrapped_obs, _, _, _, _ = wrapped_env.step(action_clipped_env)

    assert np.alltrue(obs == wrapped_obs)


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            GenericTestEnv(action_space=DICT_SPACE, step_fn=env_step_fn),
            {"box": (NEW_BOX_LOW, NEW_BOX_HIGH)},
            {"box": NEW_BOX_HIGH + 1, "box2": 0, "discrete": 0},
        )
    ],
)
def test_clip_actions_v0_dict_action(env, args, action):
    """Checks `Dict` action spaces clipping.

    Check whether actions in `Dict` action spaces are
    correctly clipped.
    """
    wrapped_env = ClipActionV0(env, args)
    _, _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["box"] == NEW_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (  # one level of nested dict
            GenericTestEnv(action_space=NESTED_DICT_SPACE, step_fn=env_step_fn),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "nested": {"nested": (NEW_BOX_LOW, NEW_BOX_HIGH)},
            },
            {
                "box": NEW_BOX_HIGH + 1,
                "discrete": 0,
                "nested": {"nested": NEW_BOX_HIGH + 1},
            },
        ),
        (  # two levels of nested dict
            GenericTestEnv(action_space=DOUBLY_NESTED_DICT_SPACE, step_fn=env_step_fn),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "nested": {"nested": {"nested": (NEW_BOX_LOW, NEW_BOX_HIGH)}},
            },
            {
                "box": NEW_BOX_HIGH + 1,
                "discrete": 0,
                "nested": {"nested": {"nested": NEW_BOX_HIGH + 1}},
            },
        ),
    ],
)
def test_clip_actions_v0_nested_dict_action(env, args, action):
    """Checks nested `Dict` action spaces clipping.

    Check whether actions in nested `Dict` action spaces are
    correctly clipped.
    """
    wrapped_env = ClipActionV0(env, args)
    _, _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    nested_action = executed_actions["nested"]
    while isinstance(nested_action, OrderedDict):
        nested_action = nested_action["nested"]

    assert executed_actions["box"] == NEW_BOX_HIGH
    assert nested_action == NEW_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            GenericTestEnv(action_space=TUPLE_SPACE, step_fn=env_step_fn),
            [None, (NEW_BOX_LOW, NEW_BOX_HIGH)],
            [0, NEW_BOX_HIGH + 1],
        )
    ],
)
def test_clip_actions_v0_tuple_action(env, args, action):
    """Checks `Tuple` action spaces clipping.

    Check whether actions in `Tuple` action spaces are
    correctly clipped.
    """
    wrapped_env = ClipActionV0(env, args)
    _, _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert np.alltrue(executed_actions == (0, NEW_BOX_HIGH))


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (  # one level of nesting
            GenericTestEnv(action_space=NESTED_TUPLE_SPACE, step_fn=env_step_fn),
            [
                (NEW_BOX_LOW, NEW_BOX_HIGH),
                [None, (NEW_BOX_LOW, NEW_BOX_HIGH)],
            ],
            [NEW_BOX_HIGH + 1, [0, NEW_BOX_HIGH + 1]],
        ),
        (  # two levels of nesting
            GenericTestEnv(action_space=DOUBLY_NESTED_TUPLE_SPACE, step_fn=env_step_fn),
            [
                (NEW_BOX_LOW, NEW_BOX_HIGH),
                [None, [None, (NEW_BOX_LOW, NEW_BOX_HIGH)]],
            ],
            [NEW_BOX_HIGH + 1, [0, [0, (NEW_BOX_HIGH + 1)]]],
        ),
    ],
)
def test_clip_actions_v0_nested_tuple_action(env, args, action):
    """Checks nested `Tuple` action spaces clipping.

    Check whether actions in nested `Tuple` action spaces are
    correctly clipped.
    """
    wrapped_env = ClipActionV0(env, args)
    _, _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    nested_action = executed_actions[-1]
    while isinstance(nested_action, tuple):
        nested_action = nested_action[-1]

    assert executed_actions[0] == NEW_BOX_HIGH
    assert nested_action == NEW_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            GenericTestEnv(action_space=DICT_WITHIN_TUPLE_SPACE, step_fn=env_step_fn),
            [None, {"dict": (NEW_BOX_LOW, NEW_BOX_HIGH)}],
            [0, {"dict": NEW_BOX_HIGH + 1}],
        )
    ],
)
def test_clip_actions_v0_dict_within_tuple(env, args, action):
    """Checks `Dict` within `Tuple` action spaces clipping.

    Check whether actions performed in `Dict` action space
    within an outer `Tuple` action space is correctly clipped.
    """
    wrapped_env = ClipActionV0(env, args)
    _, _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions[1]["dict"] == NEW_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            GenericTestEnv(action_space=TUPLE_WITHIN_DICT_SPACE, step_fn=env_step_fn),
            {"tuple": [(NEW_BOX_LOW, NEW_BOX_HIGH)]},
            {"discrete": 0, "tuple": [NEW_BOX_HIGH + 1]},
        )
    ],
)
def test_clip_actions_v0_tuple_within_dict(env, args, action):
    """Checks Tuple within Dict action spaces clipping.

    Check whether actions performed in `Tuple` action space
    within an outer `Dict` action space is correctly clipped.
    """
    wrapped_env = ClipActionV0(env, args)
    _, _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["tuple"][0] == NEW_BOX_HIGH
