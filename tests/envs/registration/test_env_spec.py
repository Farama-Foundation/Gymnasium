"""Test for the `EnvSpec`, in particular, a full integration with `EnvSpec`."""

from __future__ import annotations

import re
from typing import Any

import dill as pickle
import pytest

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import check_env, data_equivalence


def test_full_integration():
    # Create an environment to test with
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    env = gym.wrappers.TimeAwareObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.8)

    # Generate the spec_stack
    env_spec = env.spec
    assert isinstance(env_spec, EnvSpec)
    # additional_wrappers = (TimeAwareObservation, NormalizeReward)
    assert len(env_spec.additional_wrappers) == 2
    # env_spec.pprint()

    # Serialize the spec_stack
    env_spec_json = env_spec.to_json()
    assert isinstance(env_spec_json, str)

    # Deserialize the spec_stack
    recreate_env_spec = EnvSpec.from_json(env_spec_json)
    # recreate_env_spec.pprint()

    assert env_spec.additional_wrappers == recreate_env_spec.additional_wrappers
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
    env = gym.wrappers.TimeAwareObservation(env)

    env_spec = env.spec
    assert env_spec is not None

    output = env_spec.pprint(disable_print=True)
    assert (
        output
        == """id=CartPole-v1
reward_threshold=475.0
max_episode_steps=500
additional_wrappers=[
	name=TimeAwareObservation, kwargs={'flatten': True, 'normalize_time': False, 'dict_time_key': 'time'}
]"""
    )

    output = env_spec.pprint(disable_print=True, include_entry_points=True)
    assert (
        output
        == """id=CartPole-v1
entry_point=gymnasium.envs.classic_control.cartpole:CartPoleEnv
reward_threshold=475.0
max_episode_steps=500
additional_wrappers=[
	name=TimeAwareObservation, entry_point=gymnasium.wrappers.stateful_observation:TimeAwareObservation, kwargs={'flatten': True, 'normalize_time': False, 'dict_time_key': 'time'}
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
disable_env_checker=False
additional_wrappers=[
	name=TimeAwareObservation, kwargs={'flatten': True, 'normalize_time': False, 'dict_time_key': 'time'}
]"""
    )

    env_spec.additional_wrappers = ()
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
disable_env_checker=False
additional_wrappers=[]"""
    )


class Unpickleable:
    def __getstate__(self):
        raise RuntimeError("Cannot pickle me!")


class EnvWithUnpickleableObj(gym.Env):
    def __init__(self, unpickleable_obj):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)

        self.unpickleable_obj = unpickleable_obj

    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.observation_space.seed(seed)
        return self.observation_space.sample(), {}


def test_spec_with_unpickleable_object():
    gym.register(
        id="TestEnv-v0",
        entry_point=EnvWithUnpickleableObj,
        kwargs={},
    )

    env = gym.make("TestEnv-v0", unpickleable_obj=Unpickleable())
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "An exception occurred (Cannot pickle me!) while copying the environment spec="
        ),
    ):
        env.spec

    check_env(env, skip_render_check=True)
    env.close()

    del gym.registry["TestEnv-v0"]
