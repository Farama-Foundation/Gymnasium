"""Test suite for NormalizeRewardV0."""
from gymnasium.experimental.wrappers import NormalizeRewardV0
from tests.testing_env import GenericTestEnv


def test_running_mean_normalize_reward_wrapper():
    """Tests that the property `_update_running_mean` freezes/continues the running statistics updating."""
    env = GenericTestEnv()
    wrapped_env = NormalizeRewardV0(env)

    # Default value is True
    assert wrapped_env.update_running_mean

    wrapped_env.reset()
    rms_var_init = wrapped_env.rewards_running_means.var
    rms_mean_init = wrapped_env.rewards_running_means.mean

    # Statistics are updated when env.step()
    wrapped_env.step(None)
    rms_var_updated = wrapped_env.rewards_running_means.var
    rms_mean_updated = wrapped_env.rewards_running_means.mean
    assert rms_var_init != rms_var_updated
    assert rms_mean_init != rms_mean_updated

    # Assure property is set
    wrapped_env.update_running_mean = False
    assert not wrapped_env.update_running_mean

    # Statistics are frozen
    wrapped_env.step(None)
    assert rms_var_updated == wrapped_env.rewards_running_means.var
    assert rms_mean_updated == wrapped_env.rewards_running_means.mean
