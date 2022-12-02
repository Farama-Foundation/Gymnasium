import numpy as np

import gymnasium as gym
from gymnasium.experimental.wrappers import DelayObservationV0

SEED = 42

DELAY = 3
NUM_STEPS = 5


def test_delay_observation():
    env = gym.make("CartPole-v1")
    env.action_space.seed(SEED)
    env.reset(seed=SEED)

    undelayed_observations = []
    for _ in range(NUM_STEPS):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        undelayed_observations.append(obs)

    env.action_space.seed(SEED)
    env.reset(seed=SEED)
    env = DelayObservationV0(env, delay=DELAY)

    delayed_observations = []
    for i in range(NUM_STEPS):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        if i < DELAY - 1:
            assert np.all(obs == 0)
        delayed_observations.append(obs)

    assert np.alltrue(
        np.array(delayed_observations[DELAY:])
        == np.array(undelayed_observations[: DELAY - 1])
    )
