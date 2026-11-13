"""Test suite for DiscretizeAction wrapper."""

import numpy as np
import pytest

from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.wrappers import DiscretizeAction
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import record_action_step


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
    for _ in range(1000):
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


def test_discretize_action_multidimensional_box():
    """DiscretizeAction should accept any finite Box, not only 1-D.

    ``n_dims`` was taken from ``shape[0]``, so a Box of shape ``(2, 2)``
    produced Discrete(4) instead of Discrete(16), reconstructed actions had
    the wrong shape, and a scalar Box raised IndexError.
    """
    env = GenericTestEnv(action_space=Box(0, 1, shape=(2, 2), dtype=np.float32))
    wrapped = DiscretizeAction(env, bins=2)
    assert wrapped.action_space == Discrete(16)
    for i in range(wrapped.action_space.n):
        act = wrapped.action(i)
        assert np.shape(act) == (2, 2)
        assert env.action_space.contains(act)
        assert wrapped.revert_action(act) == i


def test_discretize_action_scalar_box():
    """A 0-D Box action space should discretize without IndexError."""
    env = GenericTestEnv(action_space=Box(0, 1, shape=(), dtype=np.float32))
    wrapped = DiscretizeAction(env, bins=2)
    assert wrapped.action_space == Discrete(2)
    act = wrapped.action(0)
    assert env.action_space.contains(act)
    assert wrapped.revert_action(act) == 0


def test_discretize_action_multidimensional_box_multidiscrete():
    """multidiscrete=True on a (2, 2) Box should yield MultiDiscrete of length 4."""
    env = GenericTestEnv(action_space=Box(0, 1, shape=(2, 2), dtype=np.float32))
    wrapped = DiscretizeAction(env, bins=2, multidiscrete=True)
    assert wrapped.action_space == MultiDiscrete([2, 2, 2, 2])
    assert len(wrapped.action_space) == 4
    for _ in range(16):
        act_discrete = wrapped.action_space.sample()
        act = wrapped.action(act_discrete)
        assert np.shape(act) == (2, 2)
        assert env.action_space.contains(act)
        assert np.all(wrapped.revert_action(act) == act_discrete)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("multidiscrete", [False, True])
def test_discretize_action_large_finite_bounds(dtype, sign, multidiscrete):
    """Large finite bin endpoints must not overflow when computing their midpoint."""
    limit = np.finfo(dtype).max
    bounds = sorted([sign * limit / 2, sign * limit])
    env = GenericTestEnv(
        action_space=Box(*bounds, shape=(2,), dtype=dtype),
        step_func=record_action_step,
    )
    with np.errstate(over="raise", invalid="raise"):
        wrapped = DiscretizeAction(env, bins=(2, 3), multidiscrete=multidiscrete)
        wrapped.reset(seed=0)

        for first_bin in range(2):
            for second_bin in range(3):
                discrete_action = (
                    np.array([first_bin, second_bin])
                    if multidiscrete
                    else first_bin * 3 + second_bin
                )
                _, _, _, _, info = wrapped.step(discrete_action)
                continuous_action = info["action"]
                assert continuous_action in env.action_space
                assert continuous_action.dtype == dtype
                expected = bounds[0] / limit + (
                    (bounds[1] / limit - bounds[0] / limit)
                    * np.array([(first_bin + 0.5) / 2, (second_bin + 0.5) / 3])
                )
                np.testing.assert_allclose(continuous_action / limit, expected)
                np.testing.assert_array_equal(
                    wrapped.revert_action(continuous_action), discrete_action
                )


@pytest.mark.parametrize("multidiscrete", [False, True])
def test_discretize_action_overflowing_range(multidiscrete):
    """A finite float64 interval may have a width larger than float64 can represent."""
    limit = np.finfo(np.float64).max
    env = GenericTestEnv(
        action_space=Box(-limit, limit, shape=(1,), dtype=np.float64),
        step_func=record_action_step,
    )
    with np.errstate(over="raise", invalid="raise"):
        wrapped = DiscretizeAction(env, bins=4, multidiscrete=multidiscrete)
        wrapped.reset(seed=0)
        for index in range(4):
            action = np.array([index]) if multidiscrete else index
            _, _, _, _, info = wrapped.step(action)
            assert info["action"] in env.action_space
            np.testing.assert_allclose(info["action"] / limit, [-0.75 + index / 2])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("bounds", [(-2.0, 2.0), (0.0, 1.0), (-0.3, 0.7)])
def test_discretize_action_preserves_finite_midpoints(dtype, bounds):
    """The overflow fallback must preserve the existing rounding on ordinary bounds."""
    env = GenericTestEnv(action_space=Box(*bounds, shape=(1,), dtype=dtype))
    wrapped = DiscretizeAction(env, bins=7)
    edges = np.linspace(env.action_space.low[0], env.action_space.high[0], 8)
    np.testing.assert_array_equal(
        wrapped.bin_centers[0], 0.5 * (edges[:-1] + edges[1:])
    )
