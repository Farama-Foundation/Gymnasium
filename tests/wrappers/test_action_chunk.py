"""Tests for distinct action sequences, episode boundaries, and vector chunk metadata."""

from contextlib import closing
from copy import deepcopy
from functools import partial

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import (
    ActionChunk,
    Autoreset,
    FrameStackObservation,
    RecordEpisodeStatistics,
    TimeLimit,
)
from tests.testing_env import GenericTestEnv


class ChunkTestEnv(gym.Env):
    """Expose executed actions, reset arguments, and controllable inner-step failures."""

    def __init__(self, shape=(1,), end=100, truncated=False, mutate=False):
        self.action_space = Box(-10, 10, shape=shape, dtype=np.float32)
        self.observation_space = Box(-10, 10, shape=shape, dtype=np.float32)
        self.end, self.truncated, self.mutate = end, truncated, mutate
        self.actions = []
        self.info = {}
        self.failure = None
        self.reward_override = None
        self.collision = None

    def reset(self, *, seed=None, options=None):
        """Record reset passthrough and start a fresh action history."""
        super().reset(seed=seed)
        self.reset_args = (seed, options)
        self.actions = []
        return np.zeros(self.observation_space.shape, dtype=np.float32), self.info

    def step(self, action):
        """Reward the distinct action, then optionally mutate or fail after advancing."""
        assert (
            isinstance(action, np.ndarray) and action.shape == self.action_space.shape
        )
        self.actions.append(action.copy())
        observation = action.copy()
        reward = (
            float(np.sum(action))
            if self.reward_override is None
            else self.reward_override
        )
        if self.mutate:
            action[...] = -10
        if self.failure is not None:
            raise self.failure
        count = len(self.actions)
        self.info = {"step": count}
        if self.collision is not None:
            self.info[self.collision] = "already used"
        ended = count >= self.end
        return (
            observation,
            reward,
            ended and not self.truncated,
            ended and self.truncated,
            self.info,
        )


@pytest.mark.parametrize("shape", [(), (1,), (2, 2)])
def test_distinct_actions_and_ownership(shape):
    """Execute each action once, preserving caller arrays and stable returned metadata."""
    base = ChunkTestEnv(shape=shape, mutate=True)
    env = ActionChunk(base, 3)
    actions = np.broadcast_to(
        np.arange(1, 4, dtype=np.float32).reshape((3,) + (1,) * len(shape)),
        (3,) + shape,
    ).copy()
    original = actions.copy()
    options = {"example": 3}
    env.reset(seed=12, options=options)
    assert base.reset_args[0] == 12 and base.reset_args[1] is options
    obs, reward, terminated, truncated, info = env.step(actions)
    np.testing.assert_array_equal(np.array(base.actions), original)
    np.testing.assert_array_equal(actions, original)
    np.testing.assert_array_equal(obs, original[-1])
    assert reward == float(original.sum()) and not (terminated or truncated)
    chunk = info["action_chunk"]
    assert chunk["length"] == 3
    assert chunk["rewards"].dtype == np.float64
    assert chunk["valid_mask"].dtype == np.bool_
    np.testing.assert_array_equal(chunk["valid_mask"], [True] * 3)
    assert "action_chunk" not in base.info
    snapshot = deepcopy(info)
    env.step(actions)
    assert data_equivalence(info, snapshot)


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("end", [1, 2, 3])
def test_early_stop_and_reset_required(end, truncated):
    base = ChunkTestEnv(end=end, truncated=truncated)
    env = ActionChunk(base, 3)
    actions = np.array([[1], [2], [3]], dtype=np.float32)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(actions)
    env.reset()
    obs, reward, terminated, was_truncated, info = env.step(actions)
    assert (terminated, was_truncated) == (not truncated, truncated)
    assert len(base.actions) == info["action_chunk"]["length"] == end
    np.testing.assert_array_equal(obs, [end])
    np.testing.assert_array_equal(
        info["action_chunk"]["rewards"], list(range(1, end + 1)) + [0] * (3 - end)
    )
    np.testing.assert_array_equal(
        info["action_chunk"]["valid_mask"], [True] * end + [False] * (3 - end)
    )
    assert reward == sum(range(1, end + 1))
    with pytest.raises(gym.error.ResetNeeded):
        env.step(actions)
    env.reset()
    env.step(actions)


@pytest.mark.parametrize(
    "action",
    [
        [[1], [2], [3]],
        np.ones((2, 1), dtype=np.float32),
        np.ones((3, 1), dtype=np.float64),
        np.array([[1], [2], [11]], dtype=np.float32),
        np.array([[1], [2], [np.nan]], dtype=np.float32),
        np.ones((3, 1), dtype=object),
    ],
)
def test_entire_chunk_validated_before_execution(action):
    base = ChunkTestEnv(end=1)
    env = ActionChunk(base, 3)
    env.reset()
    with pytest.raises((TypeError, ValueError)):
        env.step(action)
    assert not base.actions
    # Even invalid unused tail actions are rejected, but the caller can retry.
    env.step(np.zeros((3, 1), dtype=np.float32))


@pytest.mark.parametrize(
    "chunk_size,error",
    [(0, ValueError), (-2, ValueError), (True, TypeError), (2.0, TypeError)],
)
def test_invalid_chunk_size(chunk_size, error):
    with pytest.raises(error):
        ActionChunk(ChunkTestEnv(), chunk_size)


@pytest.mark.parametrize("info_key", ["", "_chunk", None, 3])
def test_invalid_info_key(info_key):
    with pytest.raises(ValueError):
        ActionChunk(ChunkTestEnv(), 2, info_key=info_key)


def test_rejected_inner_wrappers_and_spaces():
    with pytest.raises(TypeError, match="Box"):
        ActionChunk(GenericTestEnv(action_space=Discrete(2)), 2)
    for inner in [Autoreset(ChunkTestEnv()), ActionChunk(ChunkTestEnv(), 2)]:
        with pytest.raises(ValueError, match="cannot contain"):
            ActionChunk(TimeLimit(inner, 20), 2)
    with closing(SyncVectorEnv([ChunkTestEnv])) as vector:
        with pytest.raises(TypeError, match="scalar"):
            ActionChunk(vector, 2)


@pytest.mark.parametrize("key", ["chunk", "_chunk"])
def test_reserved_info_collision(key):
    base = ChunkTestEnv()
    env = ActionChunk(base, 3, info_key="chunk")
    base.info = {key: 1}
    with pytest.raises(ValueError, match="collision"):
        env.reset()
    base.info = {}
    env.reset()
    base.collision = key
    with pytest.raises(ValueError, match="collision"):
        env.step(np.zeros((3, 1), dtype=np.float32))
    assert len(base.actions) == 1
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros((3, 1), dtype=np.float32))


def test_inner_failure_and_reward_contract():
    base = ChunkTestEnv()
    env = ActionChunk(base, 2)
    action = np.zeros((2, 1), dtype=np.float32)
    env.reset()
    base.failure = RuntimeError("inner failure")
    with pytest.raises(RuntimeError, match="inner failure"):
        env.step(action)
    base.failure = None
    with pytest.raises(gym.error.ResetNeeded):
        env.step(action)
    env.reset()
    base.reward_override = np.ones(2)
    with pytest.raises(TypeError, match="scalar"):
        env.step(action)
    for reward in [np.inf, np.nan, np.float32(2.5)]:
        env.reset()
        base.reward_override = reward
        total = env.step(action)[1]
        assert np.isnan(total) if np.isnan(reward) else total == 2 * reward


def test_wrapper_order_and_outer_autoreset():
    actions = np.ones((3, 1), dtype=np.float32)
    inner_stats = ActionChunk(RecordEpisodeStatistics(TimeLimit(ChunkTestEnv(), 2)), 3)
    inner_stats.reset()
    _, _, _, truncated, info = inner_stats.step(actions)
    assert truncated and info["episode"]["l"] == 2
    outer_stats = RecordEpisodeStatistics(TimeLimit(ActionChunk(ChunkTestEnv(), 3), 2))
    outer_stats.reset()
    outer_stats.step(actions)
    _, _, _, truncated, info = outer_stats.step(actions)
    assert truncated and info["episode"]["l"] == 2
    assert len(outer_stats.unwrapped.actions) == 6
    stacked = ActionChunk(FrameStackObservation(ChunkTestEnv(), 3), 3)
    stacked.reset()
    observation, *_ = stacked.step(np.array([[1], [2], [3]], dtype=np.float32))
    np.testing.assert_array_equal(observation, [[1], [2], [3]])
    auto = Autoreset(ActionChunk(ChunkTestEnv(end=1), 3))
    auto.reset()
    auto.step(actions)
    _, reward, terminated, truncated, info = auto.step(actions)
    assert reward == 0 and not (terminated or truncated) and "action_chunk" not in info
    assert not auto.unwrapped.actions


def make_chunk_env(end):
    """Make a worker-local chunk wrapper for spawn-compatible vector tests."""
    return ActionChunk(ChunkTestEnv(end=end), 3)


@pytest.mark.parametrize("vector_class", [SyncVectorEnv, AsyncVectorEnv])
@pytest.mark.parametrize("mode", list(AutoresetMode))
def test_vector_chunk_info_and_autoreset(vector_class, mode):
    kwargs = {"context": "spawn"} if vector_class is AsyncVectorEnv else {}
    with closing(
        vector_class(
            [partial(make_chunk_env, 2), partial(make_chunk_env, 5)],
            autoreset_mode=mode,
            **kwargs,
        )
    ) as env:
        env.reset(seed=2)
        actions = np.ones((2, 3, 1), dtype=np.float32)
        _, rewards, terminated, _, info = env.step(actions)
        np.testing.assert_array_equal(rewards, [2, 3])
        np.testing.assert_array_equal(terminated, [True, False])
        if mode == AutoresetMode.SAME_STEP:
            assert info["final_info"]["action_chunk"]["length"][0] == 2
            np.testing.assert_array_equal(info["_action_chunk"], [False, True])
        else:
            np.testing.assert_array_equal(info["action_chunk"]["length"], [2, 3])
            np.testing.assert_array_equal(
                info["action_chunk"]["valid_mask"],
                [[True, True, False], [True, True, True]],
            )
        if mode == AutoresetMode.DISABLED:
            env.reset(options={"reset_mask": np.array([True, False])})
        _, _, _, _, info = env.step(actions)
        if mode == AutoresetMode.NEXT_STEP:
            np.testing.assert_array_equal(info["_action_chunk"], [False, True])


def test_spec_roundtrip():
    env = ActionChunk(gym.make("Pendulum-v1"), np.int64(3), info_key="chunk")
    try:
        reconstructed = gym.make(env.spec)
        try:
            assert reconstructed.action_space == env.action_space
            assert reconstructed._saved_kwargs == env._saved_kwargs
        finally:
            reconstructed.close()
    finally:
        env.close()


def test_pendulum_matches_manual_execution():
    """Seeded dynamics and rewards match stepping the same distinct actions by hand."""
    with (
        gym.make("Pendulum-v1") as manual,
        ActionChunk(gym.make("Pendulum-v1"), 4) as chunked,
    ):
        assert data_equivalence(manual.reset(seed=31), chunked.reset(seed=31))
        rng = np.random.default_rng(12)
        for _ in range(5):
            actions = rng.uniform(-2, 2, size=(4, 1)).astype(np.float32)
            rewards = []
            for action in actions:
                obs, reward, terminated, truncated, info = manual.step(action)
                rewards.append(reward)
            actual = chunked.step(actions)
            assert data_equivalence(actual[0], obs)
            assert actual[1] == pytest.approx(sum(rewards))
            assert actual[2:4] == (terminated, truncated)
            np.testing.assert_array_equal(actual[4]["action_chunk"]["rewards"], rewards)
            assert {
                key: value for key, value in actual[4].items() if key != "action_chunk"
            } == info
