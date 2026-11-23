"""Test suite for vector FrameStackObservation wrapper."""

from __future__ import annotations

import re
from copy import deepcopy

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces, wrappers
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.utils.passive_env_checker import data_shares_objects
from gymnasium.vector import AutoresetMode, SyncVectorEnv, VectorEnv
from gymnasium.vector.utils import batch_space, iterate
from gymnasium.wrappers.vector import FrameStackObservation
from tests.testing_env import GenericTestEnv
from tests.wrappers.vector.utils import (
    EpisodeEndTracker,
    check_vector_wrapper_equivalence,
    make_staggered_vector_env,
)


def _make_counter_env(
    obs_space: spaces.Space | None = None,
    max_steps: int = 10,
) -> GenericTestEnv:
    """Create a deterministic counter environment for exact frame-by-frame verification."""
    if obs_space is None:
        obs_space = spaces.Box(low=-1000.0, high=1000.0, shape=(3,), dtype=np.float32)

    state = {"t": 0, "episode": 0}

    def reset_func(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        state["episode"] += 1
        state["t"] = 0
        val = float(state["episode"] * 100)
        obs = np.full(obs_space.shape, val, dtype=obs_space.dtype)
        return obs, {"episode_id": state["episode"], "step": 0}

    def step_func(self, action):
        state["t"] += 1
        val = float(state["episode"] * 100 + state["t"])
        obs = np.full(obs_space.shape, val, dtype=obs_space.dtype)
        terminated = state["t"] >= max_steps
        return (
            obs,
            1.0,
            terminated,
            False,
            {"episode_id": state["episode"], "step": state["t"]},
        )

    return GenericTestEnv(
        observation_space=obs_space,
        action_space=spaces.Discrete(2),
        reset_func=reset_func,
        step_func=step_func,
    )


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("stack_size", [1, 2, 4])
@pytest.mark.parametrize("padding_type", ["reset", "zero"])
def test_equivalence_with_single_env_wrapper(
    autoreset_mode: AutoresetMode,
    num_envs: int,
    stack_size: int,
    padding_type: str,
):
    """Verify `vector.FrameStackObservation(VectorEnv)` matches `VectorEnv(FrameStackObservation)`."""
    kwargs = {"stack_size": stack_size, "padding_type": padding_type}
    check_vector_wrapper_equivalence(
        autoreset_mode=autoreset_mode,
        num_envs=num_envs,
        env_id="CartPole-v1",
        vector_wrapper=FrameStackObservation,
        vector_kwargs=kwargs,
        env_wrapper=wrappers.FrameStackObservation,
        env_kwargs=kwargs,
        num_steps=40,
    )


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
def test_custom_padding_equivalence(autoreset_mode: AutoresetMode):
    """Verify custom padding matches single-env FrameStackObservation and is isolated from caller mutations."""
    custom_pad = np.array([1.0, -1.0, 0.2, 2.0], dtype=np.float32)
    kwargs = {"stack_size": 3, "padding_type": custom_pad}
    check_vector_wrapper_equivalence(
        autoreset_mode=autoreset_mode,
        num_envs=2,
        env_id="CartPole-v1",
        vector_wrapper=FrameStackObservation,
        vector_kwargs=kwargs,
        env_wrapper=wrappers.FrameStackObservation,
        env_kwargs=kwargs,
        num_steps=35,
    )

    # Mutating the caller's padding array after construction must not affect the wrapper
    vec_env = FrameStackObservation(
        gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync"),
        stack_size=3,
        padding_type=custom_pad,
    )
    original_pad = custom_pad.copy()
    custom_pad[:] = 999.0
    obs, _ = vec_env.reset(seed=123)
    assert np.all(obs[0, 0] == original_pad)
    assert np.all(obs[0, 1] == original_pad)
    vec_env.close()


def test_padding_types_exact_values():
    """Verify exact stacked frame values for reset, zero, and custom padding."""
    raw_env = SyncVectorEnv([lambda: _make_counter_env(max_steps=5) for _ in range(2)])
    raw_reset_obs, _ = raw_env.reset(seed=123)
    raw_step_obs, *_ = raw_env.step(np.array([0, 0]))
    raw_env.close()

    # Default: padding_type="reset"
    env_reset_pad = FrameStackObservation(
        SyncVectorEnv([lambda: _make_counter_env(max_steps=5) for _ in range(2)]),
        stack_size=3,
    )
    obs, _ = env_reset_pad.reset(seed=123)
    for i in range(2):
        expected = np.stack([raw_reset_obs[i], raw_reset_obs[i], raw_reset_obs[i]])
        assert np.array_equal(obs[i], expected)
    obs, *_ = env_reset_pad.step(np.array([0, 0]))
    for i in range(2):
        expected = np.stack([raw_reset_obs[i], raw_reset_obs[i], raw_step_obs[i]])
        assert np.array_equal(obs[i], expected)
    env_reset_pad.close()

    # padding_type="zero"
    env_zero_pad = FrameStackObservation(
        SyncVectorEnv([lambda: _make_counter_env(max_steps=5) for _ in range(2)]),
        stack_size=3,
        padding_type="zero",
    )
    obs, _ = env_zero_pad.reset(seed=123)
    zeros = np.zeros(3, dtype=np.float32)
    for i in range(2):
        expected = np.stack([zeros, zeros, raw_reset_obs[i]])
        assert np.array_equal(obs[i], expected)
    obs, *_ = env_zero_pad.step(np.array([0, 0]))
    for i in range(2):
        expected = np.stack([zeros, raw_reset_obs[i], raw_step_obs[i]])
        assert np.array_equal(obs[i], expected)
    env_zero_pad.close()

    # Custom padding
    custom = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    env_custom_pad = FrameStackObservation(
        SyncVectorEnv([lambda: _make_counter_env(max_steps=5) for _ in range(2)]),
        stack_size=3,
        padding_type=custom,
    )
    obs, _ = env_custom_pad.reset(seed=123)
    for i in range(2):
        expected = np.stack([custom, custom, raw_reset_obs[i]])
        assert np.array_equal(obs[i], expected)
    obs, *_ = env_custom_pad.step(np.array([0, 0]))
    for i in range(2):
        expected = np.stack([custom, raw_reset_obs[i], raw_step_obs[i]])
        assert np.array_equal(obs[i], expected)
    env_custom_pad.close()


def test_same_step_autoreset_final_obs_and_isolation():
    """Verify SAME_STEP stacks `info['final_obs']` with terminal frame and resets only done envs."""
    env = FrameStackObservation(
        SyncVectorEnv(
            [
                lambda: _make_counter_env(max_steps=2),
                lambda: _make_counter_env(max_steps=5),
            ],
            autoreset_mode=AutoresetMode.SAME_STEP,
        ),
        stack_size=3,
        padding_type="zero",
    )
    obs, _ = env.reset(seed=0)
    # Initial: [0, 0, 100] for both envs
    assert np.all(obs[0, :, 0] == [0.0, 0.0, 100.0])
    assert np.all(obs[1, :, 0] == [0.0, 0.0, 100.0])

    # Step 1: [0, 100, 101] for both envs
    obs, _, term, trunc, info = env.step(np.array([0, 0]))
    assert not np.any(term) and not np.any(trunc)
    assert "final_obs" not in info
    assert np.all(obs[0, :, 0] == [0.0, 100.0, 101.0])
    assert np.all(obs[1, :, 0] == [0.0, 100.0, 101.0])

    # Step 2: env 0 terminates at 102 and immediately resets to 200; env 1 continues to 102
    obs, _, term, trunc, info = env.step(np.array([0, 0]))
    assert np.array_equal(term, np.array([True, False]))
    assert np.array_equal(info["_final_obs"], np.array([True, False]))
    assert info["final_obs"][1] is None
    assert info["final_obs"][0] in env.single_observation_space
    assert np.all(info["final_obs"][0][:, 0] == [100.0, 101.0, 102.0])
    # Env 0's returned obs is the fresh reset stack [0, 0, 200], env 1 is unaffected [100, 101, 102]
    assert np.all(obs[0, :, 0] == [0.0, 0.0, 200.0])
    assert np.all(obs[1, :, 0] == [100.0, 101.0, 102.0])
    env.close()


def test_next_step_autoreset_and_explicit_masked_reset():
    """Verify NEXT_STEP resets only done envs on the following step and clears pending flags on explicit reset."""
    env = FrameStackObservation(
        SyncVectorEnv(
            [
                lambda: _make_counter_env(max_steps=2),
                lambda: _make_counter_env(max_steps=5),
            ],
            autoreset_mode=AutoresetMode.NEXT_STEP,
        ),
        stack_size=3,
        padding_type="zero",
    )
    env.reset(seed=0)
    env.step(np.array([0, 0]))  # t=1: [0, 100, 101]
    obs, _, term, _, _ = env.step(np.array([0, 0]))  # t=2: env 0 done
    assert np.array_equal(term, np.array([True, False]))
    assert np.all(obs[0, :, 0] == [100.0, 101.0, 102.0])
    assert np.all(obs[1, :, 0] == [100.0, 101.0, 102.0])

    # Next step: env 0 autoresets to 200 -> [0, 0, 200]; env 1 advances to 103 -> [101, 102, 103]
    obs, _, term, _, _ = env.step(np.array([0, 0]))
    assert not np.any(term)
    assert np.all(obs[0, :, 0] == [0.0, 0.0, 200.0])
    assert np.all(obs[1, :, 0] == [101.0, 102.0, 103.0])

    # Advance env 0 to termination again (201 -> 202), then explicitly reset env 0 via reset_mask
    env.step(np.array([0, 0]))  # env0: 201, env1: 104
    _, _, term, _, _ = env.step(np.array([0, 0]))  # env0: 202 (done), env1: 105 (done)
    assert np.array_equal(term, np.array([True, True]))

    obs, _ = env.reset(options={"reset_mask": np.array([True, False])})
    # Env 0 was explicitly reset to 300; env 1 keeps its stack [103, 104, 105]
    assert np.all(obs[0, :, 0] == [0.0, 0.0, 300.0])
    assert np.all(obs[1, :, 0] == [103.0, 104.0, 105.0])

    # Next step: env 0 must NOT autoreset again (advances to 301), while env 1 autoresets to 200
    obs, _, _, _, _ = env.step(np.array([0, 0]))
    assert np.all(obs[0, :, 0] == [0.0, 300.0, 301.0])
    assert np.all(obs[1, :, 0] == [0.0, 0.0, 200.0])
    env.close()


def test_disabled_autoreset_partial_reset_mask():
    """Verify DISABLED mode with partial reset_mask only resets masked sub-environments."""
    env = FrameStackObservation(
        SyncVectorEnv(
            [
                lambda: _make_counter_env(max_steps=2),
                lambda: _make_counter_env(max_steps=5),
            ],
            autoreset_mode=AutoresetMode.DISABLED,
        ),
        stack_size=3,
        padding_type="reset",
    )
    env.reset(seed=0)
    env.step(np.array([0, 0]))  # [100, 100, 101]
    obs, _, term, _, _ = env.step(np.array([0, 0]))  # [100, 101, 102], env 0 done
    assert np.array_equal(term, np.array([True, False]))

    obs, _ = env.reset(options={"reset_mask": np.array([True, False])})
    assert np.all(obs[0, :, 0] == [200.0, 200.0, 200.0])
    assert np.all(obs[1, :, 0] == [100.0, 101.0, 102.0])

    obs, _, _, _, _ = env.step(np.array([0, 0]))
    assert np.all(obs[0, :, 0] == [200.0, 200.0, 201.0])
    assert np.all(obs[1, :, 0] == [101.0, 102.0, 103.0])
    env.close()


@pytest.mark.parametrize(
    "obs_space",
    [
        spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
        spaces.Discrete(5),
        spaces.MultiDiscrete([3, 4]),
        spaces.MultiBinary(4),
        spaces.Dict(
            {
                "pos": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "mode": spaces.Discrete(3),
            }
        ),
        spaces.Tuple(
            (
                spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                spaces.Discrete(4),
            )
        ),
    ],
    ids=[
        "Box_inf",
        "Box_uint8_image",
        "Discrete",
        "MultiDiscrete",
        "MultiBinary",
        "Dict",
        "Tuple",
    ],
)
def test_diverse_observation_spaces(obs_space: spaces.Space):
    """Verify FrameStackObservation works across Box, Discrete, MultiDiscrete, MultiBinary, Dict, and Tuple spaces."""
    num_envs = 2
    stack_size = 3

    def make_env():
        return GenericTestEnv(observation_space=deepcopy(obs_space))

    raw_env = SyncVectorEnv([make_env for _ in range(num_envs)])
    stacked_env = FrameStackObservation(
        SyncVectorEnv([make_env for _ in range(num_envs)]),
        stack_size=stack_size,
    )

    expected_single = batch_space(obs_space, n=stack_size)
    assert stacked_env.single_observation_space == expected_single
    assert stacked_env.observation_space == batch_space(expected_single, n=num_envs)

    raw_obs, _ = raw_env.reset(seed=42)
    stacked_obs, _ = stacked_env.reset(seed=42)
    assert stacked_obs in stacked_env.observation_space

    raw_history = [
        [o] * stack_size for o in iterate(raw_env.observation_space, raw_obs)
    ]
    for env_idx, stacked_single in enumerate(
        iterate(stacked_env.observation_space, stacked_obs)
    ):
        assert data_equivalence(
            raw_history[env_idx],
            list(iterate(stacked_env.single_observation_space, stacked_single)),
        )

    raw_env.action_space.seed(42)
    stacked_env.action_space.seed(42)
    for _ in range(5):
        action = stacked_env.action_space.sample()
        raw_obs, *_ = raw_env.step(action)
        stacked_obs, *_ = stacked_env.step(action)
        assert stacked_obs in stacked_env.observation_space

        for env_idx, raw_single in enumerate(
            iterate(raw_env.observation_space, raw_obs)
        ):
            raw_history[env_idx] = raw_history[env_idx][1:] + [raw_single]

        for env_idx, stacked_single in enumerate(
            iterate(stacked_env.observation_space, stacked_obs)
        ):
            assert data_equivalence(
                raw_history[env_idx],
                list(iterate(stacked_env.single_observation_space, stacked_single)),
            )

    raw_env.close()
    stacked_env.close()


def test_seed_determinism_and_no_buffer_aliasing():
    """Verify deterministic trajectories for identical seeds and no shared-buffer mutation across calls."""
    env_1 = FrameStackObservation(
        make_staggered_vector_env("CartPole-v1", 3, AutoresetMode.SAME_STEP),
        stack_size=3,
    )
    env_2 = FrameStackObservation(
        make_staggered_vector_env("CartPole-v1", 3, AutoresetMode.SAME_STEP),
        stack_size=3,
    )

    obs_1, _ = env_1.reset(seed=123)
    obs_2, _ = env_2.reset(seed=123)
    assert data_equivalence(obs_1, obs_2, exact=True)

    returned_obs = [obs_1]
    env_1.action_space.seed(123)
    tracker = EpisodeEndTracker(3)
    for _ in range(30):
        action = env_1.action_space.sample()
        o1, r1, term1, trunc1, info1 = env_1.step(action)
        o2, r2, term2, trunc2, info2 = env_2.step(deepcopy(action))

        assert data_equivalence(o1, o2, exact=True)
        assert data_equivalence(r1, r2, exact=True)
        assert data_equivalence(term1, term2, exact=True)
        assert data_equivalence(trunc1, trunc2, exact=True)
        assert data_equivalence(info1, info2, exact=True)

        for prev in returned_obs[-3:]:
            assert not data_shares_objects(prev, o1)
        returned_obs.append(o1)
        tracker.update(term1, trunc1)

    tracker.check(30)
    env_1.close()
    env_2.close()


def test_missing_and_invalid_autoreset_mode():
    """Verify warning when autoreset_mode metadata is missing and TypeError when invalid."""

    class MinimalVectorEnv(VectorEnv):
        def __init__(self, metadata: dict):
            self.metadata = metadata
            self.num_envs = 2
            self.single_observation_space = spaces.Box(0.0, 1.0, (2,), dtype=np.float32)
            self.observation_space = batch_space(self.single_observation_space, 2)
            self.single_action_space = spaces.Discrete(2)
            self.action_space = batch_space(self.single_action_space, 2)

    with pytest.warns(UserWarning, match="missing `autoreset_mode`"):
        wrapped = FrameStackObservation(MinimalVectorEnv({}), stack_size=2)
        assert wrapped._autoreset_mode == AutoresetMode.NEXT_STEP

    with pytest.raises(TypeError, match="Expected env.metadata\\['autoreset_mode'\\]"):
        FrameStackObservation(
            MinimalVectorEnv({"autoreset_mode": "NextStep"}), stack_size=2
        )


def test_invalid_arguments():
    """Verify TypeError and ValueError on invalid stack_size or padding_type."""
    env = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The stack_size is expected to be an integer, actual type: <class 'float'>"
        ),
    ):
        FrameStackObservation(env, stack_size=1.0)  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The stack_size is expected to be an integer, actual type: <class 'bool'>"
        ),
    ):
        FrameStackObservation(env, stack_size=True)  # type: ignore[arg-type]

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
            "The stack_size needs to be greater than zero, actual value: -2"
        ),
    ):
        FrameStackObservation(env, stack_size=-2)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: 'unknown'"
        ),
    ):
        FrameStackObservation(env, stack_size=3, padding_type="unknown")

    invalid_padding = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    with pytest.raises(
        ValueError,
        match="Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space",
    ):
        FrameStackObservation(env, stack_size=3, padding_type=invalid_padding)

    env.close()
