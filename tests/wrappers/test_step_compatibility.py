from functools import partial

import pytest

import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility
from tests.generic_test_env import GenericTestEnv, old_step_fn

OldStepEnv = partial(GenericTestEnv, step_fn=old_step_fn)
NewStepEnv = GenericTestEnv


@pytest.mark.parametrize("env", [OldStepEnv, NewStepEnv])
@pytest.mark.parametrize("output_truncation_bool", [None, True])
def test_step_compatibility_to_new_api(env, output_truncation_bool):
    if output_truncation_bool is None:
        env = StepAPICompatibility(env())
    else:
        env = StepAPICompatibility(env(), output_truncation_bool)
    step_returns = env.step(0)
    _, _, terminated, truncated, _ = step_returns
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


@pytest.mark.parametrize("env", [OldStepEnv, NewStepEnv])
def test_step_compatibility_to_old_api(env):
    env = StepAPICompatibility(env(), False)
    step_returns = env.step(0)
    assert len(step_returns) == 4
    _, _, done, _ = step_returns
    assert isinstance(done, bool)


@pytest.mark.parametrize("apply_api_compatibility", [None, True, False])
def test_step_compatibility_in_make(apply_api_compatibility):
    gym.register("OldStepEnv-v0", entry_point=OldStepEnv)

    if apply_api_compatibility is not None:
        env = gym.make(
            "OldStepEnv-v0",
            apply_api_compatibility=apply_api_compatibility,
            disable_env_checker=True,
        )
    else:
        env = gym.make("OldStepEnv-v0", disable_env_checker=True)

    env.reset()
    step_returns = env.step(0)
    if apply_api_compatibility:
        assert len(step_returns) == 5
        _, _, terminated, truncated, _ = step_returns
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    else:
        assert len(step_returns) == 4
        _, _, done, _ = step_returns
        assert isinstance(done, bool)

    gym.envs.registry.pop("OldStepEnv-v0")
