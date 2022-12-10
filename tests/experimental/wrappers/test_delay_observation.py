"""Test suite for DelayObservationV0."""
import numpy as np

import gymnasium as gym
from gymnasium.experimental.wrappers import DelayObservationV0
from tests.experimental.wrappers.utils import DELAY, NUM_STEPS, SEED


def test_delay_observation_wrapper():
    """Tests the delay observation wrapper."""
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
