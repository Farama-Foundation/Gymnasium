"""Example file showing usage of env.specstack."""
import json

import pytest

import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import data_equivalence


def test_full_integration():
    # Create an environment to test with
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.TimeAwareObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.8)

    # Generate the spec_stack
    env_spec = env.spec
    assert isinstance(env_spec, EnvSpec)
    # env_spec.pprint()

    # Serialize the spec_stack
    env_spec_json = env_spec.to_json()
    assert isinstance(env_spec_json, str)

    # Deserialize the spec_stack
    recreate_env_spec = EnvSpec.from_json(env_spec_json)
    # recreate_env_spec.pprint()

    for wrapper_spec, recreated_wrapper_spec in zip(
        env_spec.applied_wrappers, recreate_env_spec.applied_wrappers
    ):
        assert (
            wrapper_spec == recreated_wrapper_spec
        ), f"{wrapper_spec} - {recreated_wrapper_spec}"
    assert recreate_env_spec == env_spec

    # Recreate the environment using the spec_stack
    recreated_env = gym.make(recreate_env_spec)
    assert recreated_env.render_mode == "rgb_array"
    assert isinstance(recreated_env, gym.wrappers.NormalizeReward)
    assert recreated_env.gamma == 0.8
    assert isinstance(recreated_env.env, gym.wrappers.TimeAwareObservation)
    assert isinstance(recreated_env.unwrapped, CartPoleEnv)

    obs, info = env.reset(seed=42)
    recreated_obs, recreated_info = recreated_env.reset(seed=42)
    assert data_equivalence(obs, recreated_obs)
    assert data_equivalence(info, recreated_info)

    # Test the pprint of the spec_stack
    spec_stack_output = env_spec.pprint(disable_print=True)
    json_spec_stack_output = env_spec.pprint(disable_print=True)
    assert spec_stack_output == json_spec_stack_output


def test_env_wrapper_spec():
    pass


@pytest.mark.parametrize("env_spec", [])
def test_env_spec_to_from_json(env_spec: EnvSpec):
    json_spec = env_spec.to_json()
    recreated_env_spec = EnvSpec.from_json(json_spec)

    assert env_spec == recreated_env_spec


def test_env_spec_pprint():
    env = gym.make("CartPole-v1")
    env_spec = env.spec

    output = env_spec.pprint(disable_print=True)
    assert output == ""

    output = env_spec.pprint(disable_print=True, include_entry_points=True)
    assert output == ""

    output = env_spec.pprint(disable_print=True, print_all=True)
    assert output == ""


