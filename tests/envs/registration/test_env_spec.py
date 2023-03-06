"""Example file showing usage of env.specstack."""
import pickle

import pytest

import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import data_equivalence


def test_full_integration():
    # Create an environment to test with
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped

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
        assert wrapper_spec == recreated_wrapper_spec
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

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    (
        recreated_obs,
        recreated_reward,
        recreated_terminated,
        recreated_truncated,
        recreated_info,
    ) = recreated_env.step(action)
    assert data_equivalence(obs, recreated_obs)
    assert data_equivalence(reward, recreated_reward)
    assert data_equivalence(terminated, recreated_terminated)
    assert data_equivalence(truncated, recreated_truncated)
    assert data_equivalence(info, recreated_info)

    # Test the pprint of the spec_stack
    spec_stack_output = env_spec.pprint(disable_print=True)
    json_spec_stack_output = env_spec.pprint(disable_print=True)
    assert spec_stack_output == json_spec_stack_output


@pytest.mark.parametrize(
    "env_spec",
    [
        gym.spec("CartPole-v1"),
        gym.make("CartPole-v1").unwrapped.spec,
        gym.make("CartPole-v1").spec,
        gym.wrappers.NormalizeReward(gym.make("CartPole-v1")).spec,
    ],
)
def test_env_spec_to_from_json(env_spec: EnvSpec):
    json_spec = env_spec.to_json()
    recreated_env_spec = EnvSpec.from_json(json_spec)

    assert env_spec == recreated_env_spec


def test_wrapped_env_entry_point():
    def _create_env():
        _env = gym.make("CartPole-v1", render_mode="rgb_array")
        _env = gym.wrappers.FlattenObservation(_env)
        return _env

    gym.register("TestingEnv-v0", entry_point=_create_env)

    env = gym.make("TestingEnv-v0")
    env = gym.wrappers.TimeAwareObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.8)

    recreated_env = gym.make(env.spec)

    obs, info = env.reset(seed=42)
    recreated_obs, recreated_info = recreated_env.reset(seed=42)
    assert data_equivalence(obs, recreated_obs)
    assert data_equivalence(info, recreated_info)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    (
        recreated_obs,
        recreated_reward,
        recreated_terminated,
        recreated_truncated,
        recreated_info,
    ) = recreated_env.step(action)
    assert data_equivalence(obs, recreated_obs)
    assert data_equivalence(reward, recreated_reward)
    assert data_equivalence(terminated, recreated_terminated)
    assert data_equivalence(truncated, recreated_truncated)
    assert data_equivalence(info, recreated_info)

    del gym.registry["TestingEnv-v0"]


def test_pickling_env_stack():
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.TimeAwareObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.8)

    pickled_env = pickle.loads(pickle.dumps(env))

    obs, info = env.reset(seed=123)
    pickled_obs, pickled_info = pickled_env.reset(seed=123)

    assert data_equivalence(obs, pickled_obs)
    assert data_equivalence(info, pickled_info)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    (
        pickled_obs,
        pickled_reward,
        pickled_terminated,
        pickled_truncated,
        pickled_info,
    ) = pickled_env.step(action)

    assert data_equivalence(obs, pickled_obs)
    assert data_equivalence(reward, pickled_reward)
    assert data_equivalence(terminated, pickled_terminated)
    assert data_equivalence(truncated, pickled_truncated)
    assert data_equivalence(info, pickled_info)

    env.close()
    pickled_env.close()


# flake8: noqa


def test_env_spec_pprint():
    env = gym.make("CartPole-v1")
    env_spec = env.spec
    assert env_spec is not None

    output = env_spec.pprint(disable_print=True)
    assert (
        output
        == """id=CartPole-v1
reward_threshold=475.0
max_episode_steps=500
applied_wrappers=[
	name=PassiveEnvChecker, kwargs={},
	name=OrderEnforcing, kwargs={'disable_render_order_enforcing': False},
	name=TimeLimit, kwargs={'max_episode_steps': 500}
]"""
    )

    output = env_spec.pprint(disable_print=True, include_entry_points=True)
    assert (
        output
        == """id=CartPole-v1
entry_point=gymnasium.envs.classic_control.cartpole:CartPoleEnv
reward_threshold=475.0
max_episode_steps=500
applied_wrappers=[
	name=PassiveEnvChecker, entry_point=gymnasium.wrappers.env_checker:PassiveEnvChecker, kwargs={},
	name=OrderEnforcing, entry_point=gymnasium.wrappers.order_enforcing:OrderEnforcing, kwargs={'disable_render_order_enforcing': False},
	name=TimeLimit, entry_point=gymnasium.wrappers.time_limit:TimeLimit, kwargs={'max_episode_steps': 500}
]"""
    )

    output = env_spec.pprint(disable_print=True, print_all=True)
    assert (
        output
        == """id=CartPole-v1
entry_point=gymnasium.envs.classic_control.cartpole:CartPoleEnv
reward_threshold=475.0
nondeterministic=False
max_episode_steps=500
order_enforce=True
autoreset=False
disable_env_checker=False
applied_api_compatibility=False
applied_wrappers=[
	name=PassiveEnvChecker, kwargs={},
	name=OrderEnforcing, kwargs={'disable_render_order_enforcing': False},
	name=TimeLimit, kwargs={'max_episode_steps': 500}
]"""
    )

    env_spec.applied_wrappers = ()
    output = env_spec.pprint(disable_print=True)
    assert (
        output
        == """id=CartPole-v1
reward_threshold=475.0
max_episode_steps=500"""
    )

    output = env_spec.pprint(disable_print=True, print_all=True)
    assert (
        output
        == """id=CartPole-v1
entry_point=gymnasium.envs.classic_control.cartpole:CartPoleEnv
reward_threshold=475.0
nondeterministic=False
max_episode_steps=500
order_enforce=True
autoreset=False
disable_env_checker=False
applied_api_compatibility=False
applied_wrappers=[]"""
    )
