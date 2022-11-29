"""Test suite for LambdaObservationV0."""

import numpy as np

import gymnasium as gym
from gymnasium.experimental.wrappers import LambdaObservationV0
from gymnasium.spaces import Box

NUM_ENVS = 3
BOX_SPACE = Box(-5, 5, (1,), dtype=np.float64)

SEED = 42
DISCRETE_ACTION = 1


def test_lambda_observation_v0():
    """Tests lambda observation.

    Tests if function is correctly applied to environment's observation.
    """
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    obs, _, _, _, _ = env.step(DISCRETE_ACTION)

    observation_shift = 1

    env.reset(seed=SEED)
    wrapped_env = LambdaObservationV0(
        env, lambda observation: observation + observation_shift
    )
    wrapped_obs, _, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert np.alltrue(wrapped_obs == obs + observation_shift)


def test_lambda_observation_v0_within_vector():
    """Tests lambda observation in vectorized environments.

    Tests if function is correctly applied to environment's observation
    in vectorized environment.
    """
    env = gym.vector.make(
        "CarRacing-v2", continuous=False, num_envs=NUM_ENVS, asynchronous=False
    )
    env.reset(seed=SEED)
    obs, _, _, _, _ = env.step(np.array([DISCRETE_ACTION for _ in range(NUM_ENVS)]))

    observation_shift = 1

    env.reset(seed=SEED)
    wrapped_env = LambdaObservationV0(
        env, lambda observation: observation + observation_shift
    )
    wrapped_obs, _, _, _, _ = wrapped_env.step(
        np.array([DISCRETE_ACTION for _ in range(NUM_ENVS)])
    )

    assert np.alltrue(wrapped_obs == obs + observation_shift)
