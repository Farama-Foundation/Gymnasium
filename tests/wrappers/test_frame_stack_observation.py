"""Test suite for FrameStackObservation wrapper."""
import re

import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector.utils import iterate
from gymnasium.wrappers import FrameStackObservation
from tests.wrappers.utils import SEED, TESTING_OBS_ENVS, TESTING_OBS_ENVS_IDS


@pytest.mark.parametrize("env", TESTING_OBS_ENVS, ids=TESTING_OBS_ENVS_IDS)
def test_env_obs(env, stack_size: int = 3):
    """Test different environment observations for testing."""
    obs, _ = env.reset(seed=SEED)
    env.action_space.seed(SEED)

    unstacked_obs = [obs for _ in range(stack_size)]
    for _ in range(stack_size * 2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        unstacked_obs.append(obs)

    env = FrameStackObservation(env, stack_size=stack_size)
    env.action_space.seed(seed=SEED)

    obs, _ = env.reset(seed=SEED)
    stacked_obs = [obs]
    assert obs in env.observation_space

    for i in range(stack_size * 2):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        stacked_obs.append(obs)
        assert obs in env.observation_space

    assert len(unstacked_obs) == len(stacked_obs) + stack_size - 1
    for i in range(len(stacked_obs)):
        assert data_equivalence(
            unstacked_obs[i : i + stack_size],
            list(iterate(env.observation_space, stacked_obs[i])),
        )


@pytest.mark.parametrize("stack_size", [2, 3, 4])
def test_stack_size(stack_size: int):
    """Test different stack sizes for FrameStackObservation wrapper."""
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed=SEED)

    # Perform a series of actions and store the resulting observations
    unstacked_obs = []
    obs, _ = env.reset(seed=SEED)
    unstacked_obs.append(obs)
    first_obs = obs  # Store the first observation
    for _ in range(5):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        unstacked_obs.append(obs)

    env = FrameStackObservation(env, stack_size=stack_size)
    env.action_space.seed(seed=SEED)

    # Perform the same series of actions and store the resulting stacked observations
    stacked_obs = []
    obs, _ = env.reset(seed=SEED)
    stacked_obs.append(obs)
    for _ in range(5):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        stacked_obs.append(obs)

    # Check that the frames in each stacked observation match the corresponding observations
    for i in range(len(stacked_obs)):
        frames = list(iterate(env.observation_space, stacked_obs[i]))
        for j in range(stack_size):
            if i - j < 0:
                expected_obs = first_obs  # Use the first observation instead of a zero observation
            else:
                expected_obs = unstacked_obs[i - j]
            assert data_equivalence(expected_obs, frames[stack_size - 1 - j])


def test_stack_size_failures():
    """Test the error raised by the FrameStackObservation."""
    env = gym.make("CartPole-v1")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The stack_size is expected to be an integer, actual type: <class 'float'>"
        ),
    ):
        FrameStackObservation(env, stack_size=1)

    with pytest.raises(
        ValueError,
        match=re.escape("The stack_size needs to be greater than one, actual value: 0"),
    ):
        FrameStackObservation(env, stack_size=0)
