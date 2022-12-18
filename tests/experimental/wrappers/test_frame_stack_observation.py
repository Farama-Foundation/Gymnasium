"""Test suite for FrameStackObservationV0."""
import re

import pytest
from gymnasium.utils.env_checker import data_equivalence

import gymnasium as gym
from gymnasium.vector.utils import create_empty_array, iterate, batch_space
from tests.experimental.wrappers.utils import complex_testing_obs_envs, complex_testing_env_ids, SEED
from gymnasium.experimental.wrappers import FrameStackObservationV0


@pytest.mark.parametrize("env", complex_testing_obs_envs, ids=complex_testing_env_ids)
def test_env_obs(env, stack_size=3):
    unstacked_obs = list(iterate(batch_space(env.observation_space, n=stack_size-1),
                                 create_empty_array(env.observation_space, n=stack_size-1)))

    obs, _ = env.reset(seed=SEED)
    unstacked_obs.append(obs)

    env.action_space.seed(SEED)
    for _ in range(stack_size*2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        unstacked_obs.append(obs)

    env = FrameStackObservationV0(env, stack_size=stack_size)
    stacked_obs = []

    obs, _ = env.reset(seed=SEED)
    stacked_obs.append(obs)

    env.action_space.seed(SEED)
    for i in range(stack_size*2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        stacked_obs.append(obs)

    assert len(unstacked_obs) == len(stacked_obs) + stack_size - 1
    for i in range(len(stacked_obs)):
        assert data_equivalence(unstacked_obs[i:i+stack_size],
                                list(iterate(env.observation_space, stacked_obs[i])))


@pytest.mark.parametrize("stack_size", [2, 3, 4])
def test_stack_size(stack_size: int):
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed=SEED)
    first_obs, _ = env.reset(seed=SEED)
    second_obs, _, _, _, _ = env.step(env.action_space.sample())

    zero_obs = next(iterate(env.observation_space, create_empty_array(env.observation_space)))

    env = FrameStackObservationV0(env, stack_size=stack_size)

    env.action_space.seed(seed=SEED)
    obs, _ = env.reset(seed=SEED)
    unstacked_obs = list(iterate(env.observation_space, obs))
    assert data_equivalence([zero_obs for _ in range(stack_size-1)], unstacked_obs[:-1])
    assert data_equivalence(first_obs, unstacked_obs[-1])

    obs, _, _, _, _ = env.step(env.action_space.sample())
    unstacked_obs = list(iterate(env.observation_space, obs))
    assert data_equivalence(second_obs, unstacked_obs[-1])


def test_stack_size_failures():
    env = gym.make("CartPole-v1")

    with pytest.raises(TypeError, match=re.escape("The stack_size is expected to be an integer, actual type: <class 'float'>")):
        FrameStackObservationV0(env, stack_size=1.0)

    with pytest.raises(ValueError, match=re.escape("The stack_size needs to be greater than one, actual value: 0")):
        FrameStackObservationV0(env, stack_size=0)
