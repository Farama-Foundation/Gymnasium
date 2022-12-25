"""Test suite for DtypeObservationV0."""
import numpy as np

from gymnasium.experimental.wrappers import DtypeObservationV0
from tests.experimental.wrappers.utils import (
    record_random_obs_reset,
    record_random_obs_step,
)
from tests.testing_env import GenericTestEnv


def test_dtype_observation():
    """Test ``DtypeObservation`` that the dtype is corrected modified."""
    env = GenericTestEnv(
        reset_func=record_random_obs_reset, step_func=record_random_obs_step
    )
    wrapped_env = DtypeObservationV0(env, dtype=np.uint8)

    obs, info = wrapped_env.reset()
    assert obs.dtype != info["obs"].dtype
    assert obs.dtype == np.uint8

    obs, _, _, _, info = wrapped_env.step(None)
    assert obs.dtype != info["obs"].dtype
    assert obs.dtype == np.uint8
