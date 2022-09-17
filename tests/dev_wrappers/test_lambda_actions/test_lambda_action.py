"""Test suite for LambdaActionV0."""
import numpy as np
import pytest

import gymnasium
from gymnasium.error import InvalidAction
from gymnasium.wrappers import LambdaActionV0
from tests.dev_wrappers.mock_data import BOX_SPACE, NUM_ENVS
from tests.dev_wrappers.utils import TestingEnv


@pytest.mark.parametrize(
    ("env", "fn", "action"),
    [
        (
            TestingEnv(action_space=BOX_SPACE),
            lambda action, _: action.astype(np.int32),
            np.float64(10),
        ),
    ],
)
def test_lambda_action_v0(env, fn, action):
    """Tests lambda action.

    Tests if function is correctly applied to environment's action.
    """
    wrapped_env = LambdaActionV0(env, fn, None)
    _, _, _, _, info = wrapped_env.step(action)
    executed_action = info["action"]

    assert isinstance(executed_action, type(fn(action, None)))


@pytest.mark.parametrize(
    ("env", "fn", "action"),
    [
        (
            gymnasium.vector.make(
                "CarRacing-v2", continuous=False, num_envs=NUM_ENVS, asynchronous=False
            ),
            lambda action, _: action.astype(np.int32),
            [np.float64(1.2) for _ in range(NUM_ENVS)],
        ),
    ],
)
def test_lambda_action_v0_within_vector(env, fn, action):
    """Tests lambda action in vectorized environments.

    Tests if function is correctly applied to environment's action
    in vectorized environment.
    """
    wrapped_env = LambdaActionV0(env, fn, [None for _ in range(NUM_ENVS)])
    wrapped_env.reset()

    wrapped_env.step(action)

    # unwrapped env should raise exception because it does not
    # support float actions
    with pytest.raises(InvalidAction):
        env.step(action)
