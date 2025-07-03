"""Test suite for DiscretizeAction wrapper."""

import numpy as np
import pytest

from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import DiscretizeAction
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
def test_discretize_action_space_uniformity(dimensions):
    """Tests that the Box action space is discretized uniformly."""
    env = GenericTestEnv(action_space=Box(0, 99, shape=(dimensions,)))
    n_bins = 7
    env = DiscretizeAction(env, n_bins)
    env_act = np.meshgrid(*(np.linspace(0, 99, n_bins) for _ in range(dimensions)))
    env_act = np.concatenate([o.flatten()[None] for o in env_act], 0).T
    env_act_discretized = np.sort([env.revert_action(a) for a in env_act])
    assert env_act.shape[0] == env.action_space.n
    assert np.all(env_act_discretized == np.arange(env.action_space.n))


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
def test_revert_discretize_action_space(dimensions, bins, multidiscrete):
    """Tests that the action is discretized correctly within the bins."""
    env = GenericTestEnv(action_space=Box(0, 99, shape=(dimensions,)))
    env_discrete = DiscretizeAction(env, bins, multidiscrete)
    for i in range(1000):
        act_discrete = env_discrete.action_space.sample()
        act_continuous = env_discrete.action(act_discrete)
        assert env.action_space.contains(act_continuous)
        assert np.all(env_discrete.revert_action(act_continuous) == act_discrete)


@pytest.mark.parametrize("high, low", [(0, np.inf), (-np.inf, np.inf), (-np.inf, 0)])
def test_discretize_action_bounds(high, low):
    """Tests the discretize action wrapper with spaces that should raise an error."""
    with pytest.raises((ValueError,)):
        DiscretizeAction(GenericTestEnv(action_space=Box(low, high, shape=(1,))))


def test_discretize_action_dtype():
    """Tests the discretize action wrapper with spaces that should raise an error."""
    with pytest.raises((TypeError,)):
        DiscretizeAction(GenericTestEnv(action_space=Discrete(10)))
