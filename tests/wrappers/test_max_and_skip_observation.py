"""Test suite for MaxAndSkipObservation wrapper."""

import re

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation


def test_max_and_skip_obs(skip: int = 4):
    """Test MaxAndSkipObservationV0."""
    env = gym.make("CartPole-v1")

    env = MaxAndSkipObservation(env, skip=skip)

    obs, _ = env.reset()
    assert obs in env.observation_space

    for _ in range(10):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert obs in env.observation_space

        if term or trunc:
            obs, _ = env.reset()
            assert obs in env.observation_space


def test_skip_size_failures():
    """Test the error raised by the MaxAndSkipObservation."""
    env = gym.make("CartPole-v1")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The skip is expected to be an integer, actual type: <class 'float'>"
        ),
    ):
        MaxAndSkipObservation(env, skip=1.0)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The skip value needs to be equal or greater than two, actual value: 0"
        ),
    ):
        MaxAndSkipObservation(env, skip=0)


class _CountingEnv(gym.Env):
    """Returns its step index as the observation and terminates at ``ends_at``."""

    observation_space = gym.spaces.Box(0, 255, shape=(1,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(1)

    def __init__(self, ends_at: int, offset: int = 0):
        """Builds the env.

        Args:
            ends_at: the step index at which the episode terminates
            offset: added to every observation, to tell one episode's frames from another's
        """
        self.ends_at = ends_at
        self.offset = offset
        self.t = 0

    def reset(self, *, seed=None, options=None):
        """Resets the step counter."""
        super().reset(seed=seed)
        self.t = 0
        return np.array([self.offset], dtype=np.uint8), {}

    def step(self, action):
        """Advances the counter and terminates once it reaches ``ends_at``."""
        self.t += 1
        obs = np.array([self.offset + self.t], dtype=np.uint8)
        return obs, 1.0, self.t >= self.ends_at, False, {}


@pytest.mark.parametrize("skip", [2, 3, 4, 5])
@pytest.mark.parametrize("extra_steps", [1, 2])
def test_episode_ending_mid_skip_returns_the_final_frame(skip: int, extra_steps: int):
    """An episode ending before the skip completes must still report frames from this step.

    The buffer used to be written only at the ``skip - 2`` and ``skip - 1`` indices, so a
    break before those left the previous step's frames in it.
    """
    ends_at = skip + extra_steps
    env = MaxAndSkipObservation(_CountingEnv(ends_at), skip=skip)
    env.reset(seed=0)

    obs, _, terminated, _, _ = env.step(0)
    while not terminated:
        obs, _, terminated, _, _ = env.step(0)

    assert obs[0] == ends_at


@pytest.mark.parametrize(
    "skip,extra_steps", [(2, 1), (3, 1), (3, 2), (4, 1), (4, 3), (5, 2), (5, 4)]
)
def test_time_limit_truncation_mid_skip_returns_the_final_frame(
    skip: int, extra_steps: int
):
    """A time limit cutting the skip short must report frames from this step too.

    ``step`` breaks on ``terminated or truncated``, so a ``TimeLimit`` lands on the same
    stale buffer a terminating env does, and this is the case a bootstrapping learner
    actually reads, since truncation does not mask the value of the next state.
    """
    inner = gym.wrappers.TimeLimit(
        _CountingEnv(ends_at=10_000), max_episode_steps=skip + extra_steps
    )
    env = MaxAndSkipObservation(inner, skip=skip)
    env.reset(seed=0)

    assert env.step(0)[0] == skip
    obs, reward, terminated, truncated, _ = env.step(0)

    assert obs[0] == skip + extra_steps
    assert reward == float(extra_steps)
    assert truncated and not terminated


class _ReusedBufferEnv(gym.Env):
    """Writes every observation into one array it keeps handing back."""

    observation_space = gym.spaces.Box(0, 255, shape=(1,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(1)

    def __init__(self, values: list[int]):
        """Builds the env.

        Args:
            values: the pixel value to report at each step, in order
        """
        self.values = values
        self.buffer = np.zeros(1, dtype=np.uint8)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        """Clears the shared array and the step counter."""
        super().reset(seed=seed)
        self.t = 0
        self.buffer[:] = 0
        return self.buffer, {}

    def step(self, action):
        """Overwrites the shared array in place and returns it again."""
        self.buffer[:] = self.values[self.t]
        self.t += 1
        return self.buffer, 1.0, False, False, {}


@pytest.mark.parametrize(
    "values,expected",
    [
        ([200, 200, 70, 10], 70),
        ([5, 5, 10, 3], 10),
        ([9, 9, 1, 1], 1),
        ([1, 2, 3, 4], 4),
    ],
)
def test_a_reused_observation_array_is_still_pooled_correctly(
    values: list[int], expected: int
):
    """An env that hands back one array every step must not collapse the two frames.

    The buffer is a ``numpy`` array, so writing into it copies the values rather than
    keeping a reference, and the pooled frames stay distinct. Values here fall as well as
    rise, so a collapsed pair would show up instead of hiding behind a running maximum.
    """
    env = MaxAndSkipObservation(_ReusedBufferEnv(values), skip=4)
    env.reset(seed=0)

    assert env.step(0)[0] == expected


@pytest.mark.parametrize("skip", [2, 3, 4])
def test_no_frame_carries_over_from_the_previous_episode(skip: int):
    """A new episode's first step must not max over frames the last episode left behind."""
    inner = _CountingEnv(2 * skip)
    env = MaxAndSkipObservation(inner, skip=skip)
    env.reset(seed=0)
    env.step(0)
    env.step(0)

    inner.ends_at, inner.offset = 1, 100
    env.reset(seed=0)
    obs, _, terminated, _, _ = env.step(0)

    assert terminated
    assert obs[0] == 101


@pytest.mark.parametrize("skip", [2, 3, 4, 5, 7])
def test_full_skip_still_maxes_the_last_two_frames(skip: int):
    """With no early end the result is unchanged: the max of the final two frames."""
    env = MaxAndSkipObservation(_CountingEnv(3 * skip), skip=skip)
    env.reset(seed=0)

    assert env.step(0)[0] == skip
    assert env.step(0)[0] == 2 * skip
    assert env.step(0)[0] == 3 * skip
