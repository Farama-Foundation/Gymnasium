"""Test suite for stateful observation wrappers: TimeAwareObservation, DelayObservation."""

import numpy as np

import gymnasium as gym
from gymnasium.experimental.wrappers import DelayObservationV0, TimeAwareObservationV0
from gymnasium.spaces import Box, Dict, Tuple
from tests.testing_env import GenericTestEnv


NUM_STEPS = 20
SEED = 0

DELAY = 3


def test_time_aware_observation_wrapper():
    """Tests the time aware observation wrapper."""
    # Test the environment observation space with Dict, Tuple and other
    env = GenericTestEnv(observation_space=Dict(arm_1=Box(0, 1), arm_2=Box(2, 3)))
    wrapped_env = TimeAwareObservationV0(env)
    assert isinstance(wrapped_env.observation_space, Dict)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert "time" in reset_obs and "time" in step_obs, f"{reset_obs}, {step_obs}"

    env = GenericTestEnv(observation_space=Tuple((Box(0, 1), Box(2, 3))))
    wrapped_env = TimeAwareObservationV0(env)
    assert isinstance(wrapped_env.observation_space, Tuple)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert len(reset_obs) == 3 and len(step_obs) == 3

    env = GenericTestEnv(observation_space=Box(0, 1))
    wrapped_env = TimeAwareObservationV0(env)
    assert isinstance(wrapped_env.observation_space, Dict)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert isinstance(reset_obs, dict) and isinstance(step_obs, dict)
    assert "obs" in reset_obs and "obs" in step_obs
    assert "time" in reset_obs and "time" in step_obs

    # Tests the flatten parameter
    env = GenericTestEnv(observation_space=Box(0, 1))
    wrapped_env = TimeAwareObservationV0(env, flatten=True)
    assert isinstance(wrapped_env.observation_space, Box)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert reset_obs.shape == (2,) and step_obs.shape == (2,)

    # Tests the normalize_time parameter
    env = GenericTestEnv(observation_space=Box(0, 1))
    wrapped_env = TimeAwareObservationV0(env, normalize_time=False)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert reset_obs["time"] == 100 and step_obs["time"] == 99

    env = GenericTestEnv(observation_space=Box(0, 1))
    wrapped_env = TimeAwareObservationV0(env, normalize_time=True)
    reset_obs, _ = wrapped_env.reset()
    step_obs, _, _, _, _ = wrapped_env.step(None)
    assert reset_obs["time"] == 0.0 and step_obs["time"] == 0.01


def test_delay_observation_wrapper():
    env = gym.make("CartPole-v1")
    env.action_space.seed(SEED)
    env.reset(seed=SEED)

    undelayed_observations = []
    for _ in range(NUM_STEPS):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        undelayed_observations.append(obs)

    env = DelayObservationV0(env, delay=DELAY)
    env.action_space.seed(SEED)
    env.reset(seed=SEED)

    delayed_observations = []
    for i in range(NUM_STEPS):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        delayed_observations.append(obs)
        if i < DELAY - 1:
            assert np.all(obs == 0)

    undelayed_observations = np.array(undelayed_observations)
    delayed_observations = np.array(delayed_observations)

    assert np.all(delayed_observations[DELAY:] == undelayed_observations[:-DELAY])
