"""Test suite for DiscretizeObservation wrapper."""

import numpy as np
import pytest

from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import DiscretizeObservation
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
def test_discretize_observation_space_uniformity(dimensions):
    """Tests that the Box observation space is discretized uniformly."""
    env = GenericTestEnv(observation_space=Box(0, 99, shape=(dimensions,)))
    n_bins = 7
    env = DiscretizeObservation(env, n_bins)
    env_obs = np.meshgrid(*(np.linspace(0, 99, n_bins) for _ in range(dimensions)))
    env_obs = np.concatenate([o.flatten()[None] for o in env_obs], 0).T
    env_obs_discretized = np.sort([env.observation(e) for e in env_obs])
    assert env_obs.shape[0] == env.observation_space.n
    assert np.all(env_obs_discretized == np.arange(env.observation_space.n))


@pytest.mark.parametrize(
    "dimensions, bins, multidiscrete",
    [
        (1, 3, False),
        (2, (3, 4), False),
        (3, (3, 4, 5), False),
        (1, 3, True),
        (2, (3, 4), True),
        (3, (3, 4, 5), True),
    ],
)
def test_revert_discretize_observation_space(dimensions, bins, multidiscrete):
    """Tests that the observation is discretized correctly within the bins."""
    env = GenericTestEnv(observation_space=Box(0, 99, shape=(dimensions,)))
    env_discrete = DiscretizeObservation(env, bins, multidiscrete)
    for i in range(1000):
        obs, _ = env.reset(seed=i)
        obs_discrete, _ = env_discrete.reset(seed=i)
        obs_reverted_min, obs_reverted_max = env_discrete.revert_observation(
            obs_discrete,
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
