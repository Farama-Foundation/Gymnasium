"""Example file showing usage of env.specstack."""
import numpy as np

import gymnasium as gym
from gymnasium.envs.box2d import LunarLander
from gymnasium.envs.registration import WrapperSpec, EnvSpec
from gymnasium.utils.spec_stack import serialize_spec_stack, deserialize_spec_stack, pprint_spec_stack
from gymnasium.utils.env_checker import data_equivalence


def test_full_integration():
    # Create an environment to test with
    env = gym.make("CartPole-v1")
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.TimeAwareObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.8)

    # Generate the spec_stack
    spec_stack = env.spec_stack
    assert isinstance(spec_stack, tuple)
    assert all(isinstance(spec, WrapperSpec) for spec in spec_stack[:-1])
    assert isinstance(spec_stack[-1], EnvSpec)

    # Serialize the spec_stack
    spec_stack_json = serialize_spec_stack(spec_stack)
    assert isinstance(spec_stack_json, str)
    print(spec_stack_json)

    # Deserialize the spec_stack
    recreate_spec_stack = deserialize_spec_stack(spec_stack_json)
    assert recreate_spec_stack == spec_stack

    # Recreate the environment using the spec_stack
    recreated_env = gym.make(recreate_spec_stack)
    assert isinstance(recreated_env, gym.wrappers.NormalizeReward)
    assert recreated_env.gamma == 0.8
    assert isinstance(recreated_env.env, gym.wrappers.TimeAwareObservation)
    assert isinstance(recreated_env.unwrapped, LunarLander)

    obs, info = env.reset(seed=42)
    recreated_obs, recreated_info = recreated_env.reset(seed=42)
    assert data_equivalence(obs, recreated_obs)
    assert data_equivalence(info, recreated_info)

    # Test the pprint of the spec_stack
    spec_stack_output = pprint_spec_stack(spec_stack)
    json_spec_stack_output = pprint_spec_stack(spec_stack_json)
    assert spec_stack_output == json_spec_stack_output


def test_env_wrapper_spec_stack():
    pass


def test_serialize_spec_stack():
    pass


def test_deserialize_spec_stack():
    pass


def test_pprint_spec_stack():
    pass
