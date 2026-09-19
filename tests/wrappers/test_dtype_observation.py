"""Test suite for DtypeObservation wrapper."""

import numpy as np
import pytest

from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from gymnasium.wrappers import DtypeObservation
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import record_random_obs_reset, record_random_obs_step


def test_dtype_observation():
    """Test ``DtypeObservation`` that the dtype is corrected modified."""
    env = GenericTestEnv(
        reset_func=record_random_obs_reset, step_func=record_random_obs_step
    )
    wrapped_env = DtypeObservation(env, dtype=np.uint8)

    obs, info = wrapped_env.reset()
    assert obs.dtype != info["obs"].dtype
    assert obs.dtype == np.uint8

    obs, _, _, _, info = wrapped_env.step(None)
    assert obs.dtype != info["obs"].dtype
    assert obs.dtype == np.uint8


@pytest.mark.parametrize("dtype_form", ("type", "dtype", "string"))
@pytest.mark.parametrize(
    "space,target_dtype",
    (
        (Box(-2, 2, shape=(2,), dtype=np.int16), np.float64),
        (Box(-2, 2, shape=(), dtype=np.int16), np.float64),
        (Discrete(3, start=-1), np.float64),
        (Discrete(3, start=-1), np.int32),
        (MultiDiscrete([3, 4], start=[-1, 2]), np.int32),
        (MultiBinary(3), np.int32),
    ),
)
def test_dtype_observation_descriptors(space, target_dtype, dtype_form):
    """Equivalent dtype descriptions produce matching observations and spaces."""
    dtype = {
        "type": target_dtype,
        "dtype": np.dtype(target_dtype),
        "string": np.dtype(target_dtype).name,
    }[dtype_form]
    env = GenericTestEnv(
        observation_space=space,
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    wrapped_env = DtypeObservation(env, dtype=dtype)
    assert wrapped_env.observation_space.dtype == np.dtype(target_dtype)

    obs, info = wrapped_env.reset()
    assert obs.dtype == np.dtype(target_dtype)
    assert obs.shape == space.shape
    assert np.asarray(obs) in wrapped_env.observation_space
    np.testing.assert_array_equal(obs, info["obs"])
    if isinstance(space, Discrete):
        assert isinstance(obs, np.generic)

    obs, _, _, _, info = wrapped_env.step(None)
    assert obs.dtype == np.dtype(target_dtype)
    assert obs.shape == space.shape
    assert np.asarray(obs) in wrapped_env.observation_space
    np.testing.assert_array_equal(obs, info["obs"])
    if isinstance(space, Discrete):
        assert isinstance(obs, np.generic)


@pytest.mark.parametrize(
    "space,dtype",
    [
        (Discrete(2), np.int64),
        (Discrete(3, start=-1), np.int32),
        (Discrete(4, start=10, dtype=np.int32), np.int16),
    ],
)
def test_dtype_observation_discrete_space_matches_original_support(space, dtype):
    """The wrapped Discrete space has the same closed interval as the original.

    Discrete(n, start=s) is {s, ..., s + n - 1}. The wrapper used
    ``Box(s, s + n)``, which both includes ``s + n`` and, for integer dtypes,
    samples that extra value.
    """
    env = GenericTestEnv(observation_space=space)
    wrapped_env = DtypeObservation(env, dtype=dtype)

    low = dtype(space.start)
    high = dtype(space.start + space.n - 1)
    assert wrapped_env.observation_space == Box(
        low=low, high=high, shape=(), dtype=dtype
    )

    last = dtype(space.start + space.n)
    assert last not in wrapped_env.observation_space
    assert last not in space

    for _ in range(50):
        sample = wrapped_env.observation_space.sample()
        assert sample in wrapped_env.observation_space
        assert int(sample) in space


def test_dtype_observation_multidiscrete_keeps_start():
    """Changing dtype of a MultiDiscrete must not drop a non-zero start."""
    space = MultiDiscrete([3, 4], start=[-1, 2])
    env = GenericTestEnv(observation_space=space)
    wrapped_env = DtypeObservation(env, dtype=np.int32)

    assert wrapped_env.observation_space == MultiDiscrete(
        [3, 4], start=[-1, 2], dtype=np.int32
    )

    obs, _ = wrapped_env.reset()
    assert obs in wrapped_env.observation_space
    assert obs.dtype == np.int32
