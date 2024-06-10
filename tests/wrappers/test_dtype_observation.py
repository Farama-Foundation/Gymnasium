"""Test suite for DtypeObservation wrapper."""

import numpy as np

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
