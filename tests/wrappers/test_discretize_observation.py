import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import DiscretizeObservation
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
def test_discretize_observation_space(dimensions):
    """Tests that the Box observation space is discretized uniformly."""
    env = GenericTestEnv(observation_space=Box(0, 99, shape=(dimensions,)))
    env = DiscretizeObservation(env, 13)
    env_obs = np.meshgrid(np.linspace(0, 99, 100), np.linspace(0, 99, 100))
    env_obs = np.concatenate([o.flatten(order="F")[None] for o in env_obs], 0).T
    assert env_obs.shape[0] == env.observation_space.n
    for i in range():
        assert i == env.observation(env_obs[i])


@pytest.mark.parametrize(
    "dimensions, bins",
    [
        (1, 3),
        (2, (3, 4)),
        (3, (3, 4, 5)),
        (4, (3, 4, 5, 6)),
    ],
)
def test_discretize_observation_space(dimensions, bins):
    """Tests that the observation is discretized correctly within the bins."""
    env = GenericTestEnv(observation_space=Box(0, 99, shape=(dimensions,)))
    env_discrete = DiscretizeObservation(env, bins)
    for i in range(1000):
        obs, _ = env.reset(seed=i)
        obs_discrete, _ = env_discrete.reset(seed=i)
        obs_reverted_min, obs_reverted_max = env_discrete.revert_observation(
            obs_discrete
        )
        assert np.all(obs >= obs_reverted_min) and np.all(obs <= obs_reverted_max)


@pytest.mark.parametrize("high, low", [(0, np.inf), (-np.inf, np.inf), (-np.inf, 0)])
def test_discretize_observation_bounds(high, low):
    """Tests the discretize observation wrapper with spaces that should raise an error."""
    with pytest.raises((ValueError,)):
        DiscretizeObservation(
            GenericTestEnv(observation_space=Box(low, high, shape=(1,)))
        )


def test_discretize_observation_dtype():
    """Tests the discretize observation wrapper with spaces that should raise an error."""
    with pytest.raises((TypeError,)):
        DiscretizeObservation(GenericTestEnv(observation_space=Discrete(10)))
