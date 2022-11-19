"""Test suite for LambdaActionV0."""
import numpy as np
import pytest

import gymnasium
from gymnasium.error import InvalidAction
from gymnasium.spaces import Box
from gymnasium.wrappers import LambdaActionV0
from tests.dev_wrappers.utils import TestingEnv

NUM_ENVS = 3
BOX_SPACE = Box(-5, 5, (1,), dtype=np.float64)


@pytest.mark.parametrize(
    ("env", "func", "action"),
    [
        (
            TestingEnv(action_space=BOX_SPACE),
            lambda action: action.astype(np.int32),
            np.float64(10),
        ),
    ],
)
def test_lambda_action_v0(env, func, action):
    """Tests lambda action.
    Tests if function is correctly applied to environment's action.
    """
    wrapped_env = LambdaActionV0(env, func)
    _, _, _, _, info = wrapped_env.step(action)
    executed_action = info["action"]

    assert isinstance(executed_action, type(func(action)))


@pytest.mark.parametrize(
    ("env", "func", "action"),
    [
        (
            gymnasium.vector.make(
                "CarRacing-v2", continuous=False, num_envs=NUM_ENVS, asynchronous=False
            ),
            lambda action: action.astype(np.int32),
            np.array([np.float64(1.2) for _ in range(NUM_ENVS)]),
        ),
    ],
)
def test_lambda_action_v0_within_vector(env, func, action):
    """Tests lambda action in vectorized environments.
    Tests if function is correctly applied to environment's action
    in vectorized environment.
    """
    wrapped_env = LambdaActionV0(env, func)
    wrapped_env.reset()

    wrapped_env.step(action)

    # unwrapped env should raise exception because it does not
    # support float actions
    with pytest.raises(InvalidAction):
        env.step(action)
