"""Test suite for NormalizeObservationV0."""
import numpy as np

from gymnasium.experimental.wrappers import NormalizeObservationV0
from tests.testing_env import GenericTestEnv


def test_running_mean_normalize_observation_wrapper():
    """Tests that the property `_update_running_mean` freezes/continues the running statistics updating."""
    env = GenericTestEnv()
    wrapped_env = NormalizeObservationV0(env)

    # Default value is True
    assert wrapped_env.update_running_mean

    wrapped_env.reset()
    rms_var_init = wrapped_env.obs_rms.var
    rms_mean_init = wrapped_env.obs_rms.mean

    # Statistics are updated when env.step()
    wrapped_env.step(np.array([1], dtype=np.float32))
    rms_var_updated = wrapped_env.obs_rms.var
    rms_mean_updated = wrapped_env.obs_rms.mean
    assert rms_var_init != rms_var_updated
    assert rms_mean_init != rms_mean_updated

    # Assure property is set
    wrapped_env.update_running_mean = False
    assert not wrapped_env.update_running_mean

    # Statistics are frozen
    wrapped_env.step(np.array([1], dtype=np.float32))
    assert rms_var_updated == wrapped_env.obs_rms.var
    assert rms_mean_updated == wrapped_env.obs_rms.mean
