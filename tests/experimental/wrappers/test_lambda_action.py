"""Test suite for LambdaActionV0."""
import numpy as np
import pytest

import gymnasium as gym
from gymnasium.error import InvalidAction
from gymnasium.experimental.wrappers import LambdaActionV0
from gymnasium.spaces import Box
from tests.testing_env import GenericTestEnv

NUM_ENVS = 3
BOX_SPACE = Box(-5, 5, (1,), dtype=np.float64)


def generic_step_fn(self, action):
    return 0, 0, False, False, {"action": action}


@pytest.mark.parametrize(
    ("env", "func", "action", "expected"),
    [
        (
            GenericTestEnv(action_space=BOX_SPACE, step_fn=generic_step_fn),
            lambda action: action + 2,
            1,
            3,
        ),
    ],
)
def test_lambda_action_v0(env, func, action, expected):
    """Tests lambda action.
    Tests if function is correctly applied to environment's action.
    """
    wrapped_env = LambdaActionV0(env, func)
    _, _, _, _, info = wrapped_env.step(action)
    executed_action = info["action"]

    assert executed_action == expected


def test_lambda_action_v0_within_vector():
    """Tests lambda action in vectorized environments.
    Tests if function is correctly applied to environment's action
    in vectorized environment.
    """
    env = gym.vector.make(
        "CarRacing-v2", continuous=False, num_envs=NUM_ENVS, asynchronous=False
    )
    action = np.ones(NUM_ENVS, dtype=np.float64)

    wrapped_env = LambdaActionV0(env, lambda action: action.astype(int))
    wrapped_env.reset()

    wrapped_env.step(action)

    # unwrapped env should raise exception because it does not
    # support float actions
    with pytest.raises(InvalidAction):
        env.step(action)
