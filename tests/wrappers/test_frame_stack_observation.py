"""Test suite for FrameStackObservation wrapper."""

import re

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector.utils import iterate
from gymnasium.wrappers import FrameStackObservation
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import SEED, TESTING_OBS_ENVS, TESTING_OBS_ENVS_IDS


@pytest.mark.parametrize("env", TESTING_OBS_ENVS, ids=TESTING_OBS_ENVS_IDS)
def test_different_obs_spaces(env, stack_size: int = 3):
    """Test across a large number of observation spaces to check if the FrameStack wrapper ."""
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

    for _ in range(stack_size * 2):
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
                # Use the first observation instead of a zero observation
                expected_obs = first_obs
            else:
                expected_obs = unstacked_obs[i - j]
            assert data_equivalence(expected_obs, frames[stack_size - 1 - j])


def test_padding_type():
    env = gym.make("CartPole-v1")
    reset_obs, _ = env.reset(seed=123)
    action = env.action_space.sample()
    step_obs, _, _, _, _ = env.step(action)

    stacked_env = FrameStackObservation(env, stack_size=3)  # default = "reset"
    stacked_obs, _ = stacked_env.reset(seed=123)
    assert np.all(np.stack([reset_obs, reset_obs, reset_obs]) == stacked_obs)
    stacked_obs, _, _, _, _ = stacked_env.step(action)
    assert np.all(np.stack([reset_obs, reset_obs, step_obs]) == stacked_obs)

    stacked_env = FrameStackObservation(env, stack_size=3, padding_type="zero")
    stacked_obs, _ = stacked_env.reset(seed=123)
    assert np.all(np.stack([np.zeros(4), np.zeros(4), reset_obs]) == stacked_obs)
    stacked_obs, _, _, _, _ = stacked_env.step(action)
    assert np.all(np.stack([np.zeros(4), reset_obs, step_obs]) == stacked_obs)

    stacked_env = FrameStackObservation(
        env, stack_size=3, padding_type=np.array([1, -1, 0, 2], dtype=np.float32)
    )
    stacked_obs, _ = stacked_env.reset(seed=123)
    assert np.all(
        np.stack(
            [
                np.array([1, -1, 0, 2], dtype=np.float32),
                np.array([1, -1, 0, 2], dtype=np.float32),
                reset_obs,
            ]
        )
        == stacked_obs
    )
    stacked_obs, _, _, _, _ = stacked_env.step(action)
    assert np.all(
        np.stack([np.array([1, -1, 0, 2], dtype=np.float32), reset_obs, step_obs])
        == stacked_obs
    )


@pytest.mark.parametrize("reuse_buffer", [False, True], ids=["fresh", "reused"])
@pytest.mark.parametrize("nested", [False, True], ids=["array", "nested"])
@pytest.mark.parametrize("padding_type", ["reset", "zero", "custom"])
@pytest.mark.parametrize("stack_size", [1, 3])
def test_observation_snapshots(reuse_buffer, nested, padding_type, stack_size):
    """Retain historical values even when an environment reuses observation storage."""
    buffer = np.full(2, 9, dtype=np.float32)
    observation_space: gym.spaces.Space = gym.spaces.Box(
        0, 255, shape=(2,), dtype=np.float32
    )
    if nested:
        observation_space = gym.spaces.Dict(
            {"value": gym.spaces.Tuple((observation_space,))}
        )

    def as_observation(array):
        return {"value": (array,)} if nested else array

    def get_observation():
        return as_observation(buffer if reuse_buffer else buffer.copy())

    def reset_func(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        buffer.fill(options["value"] if options else 1)
        return get_observation(), {"value": float(buffer[0])}

    def step_func(self, action):
        buffer[:] += 1
        return get_observation(), 1.0, False, False, {"value": float(buffer[0])}

    env = FrameStackObservation(
        GenericTestEnv(
            observation_space=observation_space,
            action_space=gym.spaces.Discrete(1),
            reset_func=reset_func,
            step_func=step_func,
        ),
        stack_size=stack_size,
        padding_type=(get_observation() if padding_type == "custom" else padding_type),
    )

    recorded_stacks = []
    for initial_value in (1, 10):
        obs, info = env.reset(seed=SEED, options={"value": initial_value})
        assert info == {"value": initial_value}
        assert obs in env.observation_space
        padding_value = {"reset": initial_value, "zero": 0, "custom": 9}[padding_type]
        expected_values = [padding_value] * (stack_size - 1) + [initial_value]
        recorded_stacks.append((obs, expected_values.copy()))

        for value in range(initial_value + 1, initial_value + stack_size + 2):
            obs, reward, terminated, truncated, info = env.step(0)
            assert (reward, terminated, truncated, info) == (
                1.0,
                False,
                False,
                {"value": value},
            )
            assert obs in env.observation_space
            expected_values = expected_values[1:] + [value]
            recorded_stacks.append((obs, expected_values.copy()))

    buffer.fill(255)
    for obs, expected_values in recorded_stacks:
        expected = [
            as_observation(np.full(2, value, dtype=np.float32))
            for value in expected_values
        ]
        assert data_equivalence(
            list(iterate(env.observation_space, obs)), expected, exact=True
        )


def test_stack_size_failures():
    """Test the error raised by the FrameStackObservation."""
    env = gym.make("CartPole-v1")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The stack_size is expected to be an integer, actual type: <class 'float'>"
        ),
    ):
        FrameStackObservation(env, stack_size=1.0)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The stack_size needs to be greater than zero, actual value: 0"
        ),
    ):
        FrameStackObservation(env, stack_size=0)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: 'unknown'"
        ),
    ):
        FrameStackObservation(env, stack_size=3, padding_type="unknown")

    invalid_padding = np.array([1, 2, 3, 4, 5])
    assert invalid_padding not in env.observation_space
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: array([1, 2, 3, 4, 5])"
        ),
    ):
        FrameStackObservation(env, stack_size=3, padding_type=invalid_padding)
