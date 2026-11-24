"""Tests for running statistics with empty batches."""

import numpy as np
import pytest

from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.utils import (
    RunningMeanStd,
    update_mean_var_count_from_moments,
)
from gymnasium.wrappers.vector import NormalizeObservation
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("populated", [False, True])
def test_empty_batch_preserves_statistics(shape, dtype, populated):
    """An empty batch preserves state and does not affect later updates."""
    rms = RunningMeanStd(shape=shape, dtype=dtype)
    expected = RunningMeanStd(shape=shape, dtype=dtype)
    batch = np.arange(4 * int(np.prod(shape)), dtype=dtype).reshape((4, *shape))
    if populated:
        rms.update(batch)
        expected.update(batch)

    rms.update(np.empty((0, *shape), dtype=dtype))

    np.testing.assert_array_equal(rms.mean, expected.mean)
    np.testing.assert_array_equal(rms.var, expected.var)
    assert rms.count == expected.count
    assert rms.mean.dtype == expected.mean.dtype
    assert rms.var.dtype == expected.var.dtype

    rms.update(batch + 10)
    expected.update(batch + 10)
    np.testing.assert_array_equal(rms.mean, expected.mean)
    np.testing.assert_array_equal(rms.var, expected.var)
    assert rms.count == expected.count


def test_nonempty_batch_with_zero_features_updates_count():
    """Empty feature dimensions are distinct from zero samples."""
    rms = RunningMeanStd(shape=(0,))
    rms.update(np.empty((3, 0)))
    assert rms.mean.shape == (0,)
    assert rms.var.shape == (0,)
    assert rms.count == 3 + 1e-4


@pytest.mark.parametrize("count", [0.0, 3.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_count_moments_preserve_statistics(count, dtype):
    """Undefined moments from zero samples must not poison existing state."""
    mean = np.array([1, 2], dtype=dtype)
    var = np.array([3, 4], dtype=dtype)
    batch_mean = np.full(2, np.nan, dtype=dtype)
    batch_var = np.full(2, np.nan, dtype=dtype)

    new_mean, new_var, new_count = update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, 0
    )

    np.testing.assert_array_equal(new_mean, mean)
    np.testing.assert_array_equal(new_var, var)
    assert new_count == count
    assert new_mean.dtype == dtype
    assert new_var.dtype == dtype


@pytest.mark.parametrize("epsilon", [0.0, 1e-4])
def test_running_statistics_ignore_zero_count_moments(epsilon):
    """The class moment-update API also leaves zero-count batches unchanged."""
    rms = RunningMeanStd(epsilon=epsilon)
    rms.update_from_moments(np.nan, np.nan, 0)
    assert rms.mean == 0
    assert rms.var == 1
    assert rms.count == epsilon


def test_empty_normalization_batch_does_not_poison_later_observations():
    """Direct batch normalization remains usable after an empty input batch."""
    env = NormalizeObservation(SyncVectorEnv([GenericTestEnv]))
    try:
        observations, _ = env.reset(seed=123)
        mean = env.obs_rms.mean.copy()
        var = env.obs_rms.var.copy()
        count = env.obs_rms.count

        empty = env.observations(observations[:0])

        assert empty.shape == observations[:0].shape
        np.testing.assert_array_equal(env.obs_rms.mean, mean)
        np.testing.assert_array_equal(env.obs_rms.var, var)
        assert env.obs_rms.count == count
        assert np.all(np.isfinite(env.observations(observations)))
    finally:
        env.close()
