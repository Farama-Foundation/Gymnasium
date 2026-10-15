"""Per-environment clocks, terminal observations, and scalar wrapper equivalence."""

from contextlib import closing
from copy import deepcopy
from functools import partial

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, Tuple
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv, VectorWrapper
from gymnasium.wrappers import TimeAwareObservation as ScalarTime
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.vector import TimeAwareObservation


class ClockEnv(gym.Env):
    """Deterministic observations with staggered early terminations and truncation."""

    def __init__(self, end=2, space_kind="box"):
        """Create a constant observation so clock errors cannot hide in the input."""
        self.end = end
        self.action_space = Discrete(2)
        box = Box(0, 1, shape=(1,), dtype=np.float32)
        self.observation_space = {
            "box": box,
            "dict": Dict({"z": box, "a": Discrete(2)}, sort_keys=False),
            "tuple": Tuple((box, Discrete(2))),
        }[space_kind]
        self.space_kind = space_kind
        self.t = 0

    def _obs(self):
        """Return a fresh observation with a known value."""
        box = np.array([0.5], dtype=np.float32)
        return {"box": box, "dict": {"z": box, "a": 1}, "tuple": (box, 1)}[
            self.space_kind
        ]

    def reset(self, *, seed=None, options=None):
        """Start an episode and expose the seed/options forwarding in info."""
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {"option": (options or {}).get("marker", 0)}

    def step(self, action):
        """Terminate at the configured step, leaving truncation to TimeLimit."""
        self.t += 1
        return self._obs(), float(action), self.t == self.end, False, {"step": self.t}


def make_env(end=2, space_kind="box", scalar=False, kwargs=None, limit=5):
    """Build a custom environment without a spec, inside an extra wrapper."""
    env = gym.Wrapper(TimeLimit(ClockEnv(end, space_kind), limit))
    return ScalarTime(env, **(kwargs or {})) if scalar else env


def vector_env(backend, factories, mode):
    """Use spawn for async tests to avoid depending on fork behavior."""
    kwargs = {"context": "spawn"} if backend is AsyncVectorEnv else {}
    return backend(factories, autoreset_mode=mode, **kwargs)


@pytest.mark.parametrize("backend", [SyncVectorEnv, AsyncVectorEnv])
@pytest.mark.parametrize("mode", list(AutoresetMode))
@pytest.mark.parametrize("space_kind", ["box", "dict", "tuple"])
@pytest.mark.parametrize(
    "flatten,normalize", [(False, False), (False, True), (True, False), (True, True)]
)
def test_staggered_equivalence(backend, mode, space_kind, flatten, normalize):
    """Compare whole transitions, including masked final obs, over unequal episodes."""
    kwargs = {"flatten": flatten, "normalize_time": normalize, "dict_time_key": "clock"}
    factories = [partial(make_env, end, space_kind) for end in (2, 3, 100)]
    reference_factories = [
        partial(make_env, end, space_kind, scalar=True, kwargs=kwargs)
        for end in (2, 3, 100)
    ]
    with (
        closing(vector_env(backend, factories, mode)) as base,
        closing(vector_env(backend, reference_factories, mode)) as reference,
    ):
        env = TimeAwareObservation(VectorWrapper(base), **kwargs)
        assert env.observation_space == reference.observation_space
        assert env.single_observation_space == reference.single_observation_space
        initial = env.reset(seed=[1, 2, 3])
        saved_initial = deepcopy(initial)
        assert data_equivalence(initial, reference.reset(seed=[1, 2, 3]))
        clocks = np.zeros(3, dtype=int)
        prev_dones = np.zeros(3, dtype=bool)
        for step in range(13):
            actual = env.step(np.array([0, 1, 0]))
            expected = reference.step(np.array([0, 1, 0]))
            assert data_equivalence(actual, expected)
            assert actual[0] in env.observation_space
            clocks += 1
            if mode == AutoresetMode.NEXT_STEP:
                clocks[prev_dones] = 0
            dones = actual[2] | actual[3]
            if mode == AutoresetMode.SAME_STEP:
                for i in np.flatnonzero(dones):
                    assert actual[4]["final_obs"][i] in env.single_observation_space
                clocks[dones] = 0
            np.testing.assert_array_equal(env.timesteps, clocks)
            prev_dones = dones
            # Explicit partial reset while another environment is pending reset.
            mask = (
                np.array([False, True, False]) if step == 1 else np.zeros(3, dtype=bool)
            )
            if mode == AutoresetMode.DISABLED:
                mask |= dones
            if mask.any():
                options = {"reset_mask": mask.copy(), "marker": 7}
                reset_result = env.reset(options=deepcopy(options))
                assert data_equivalence(
                    reset_result, reference.reset(options=deepcopy(options))
                )
                clocks[mask] = 0
                prev_dones[mask] = False
                np.testing.assert_array_equal(env.timesteps, clocks)
        assert data_equivalence(initial, saved_initial)
        assert data_equivalence(env.reset(seed=42), reference.reset(seed=42))
        np.testing.assert_array_equal(env.timesteps, 0)


@pytest.mark.parametrize("mode", list(AutoresetMode))
@pytest.mark.parametrize("normalize", [False, True])
def test_explicit_clock_values(mode, normalize):
    """Assert elapsed time and terminal/reset distinction independently of the scalar wrapper."""
    with closing(
        SyncVectorEnv(
            [partial(make_env, 2), partial(make_env, 100)], autoreset_mode=mode
        )
    ) as base:
        env = TimeAwareObservation(base, flatten=False, normalize_time=normalize)
        env.reset()
        first = env.step([0, 0])[0]
        terminal = env.step([0, 0])
        scale = 5 if normalize else 1
        np.testing.assert_allclose(first["time"], np.array([[1], [1]]) / scale)
        assert first["time"].dtype == (np.float32 if normalize else np.int32)
        np.testing.assert_allclose(
            terminal[0]["time"],
            np.array([[0 if mode == AutoresetMode.SAME_STEP else 2], [2]]) / scale,
        )
        if mode == AutoresetMode.SAME_STEP:
            np.testing.assert_allclose(terminal[4]["final_obs"][0]["time"], [2 / scale])
            np.testing.assert_array_equal(terminal[4]["_final_obs"], [True, False])
        elif mode == AutoresetMode.NEXT_STEP:
            np.testing.assert_allclose(
                env.step([0, 0])[0]["time"], np.array([[0], [3]]) / scale
            )
        else:
            obs, _ = env.reset(options={"reset_mask": np.array([True, False])})
            np.testing.assert_allclose(obs["time"], np.array([[0], [2]]) / scale)


@pytest.mark.parametrize(
    "mask,error",
    [
        ([True, False], TypeError),
        (np.array([True]), ValueError),
        (np.array([1, 0]), TypeError),
        (np.array([False, False]), ValueError),
    ],
)
def test_invalid_partial_reset(mask, error):
    """Failed mask validation must not reset any clocks."""
    with closing(SyncVectorEnv([make_env, make_env])) as base:
        env = TimeAwareObservation(base)
        env.reset()
        env.step([0, 0])
        with pytest.raises(error, match="reset_mask"):
            env.reset(options={"reset_mask": mask})
        np.testing.assert_array_equal(env.timesteps, [1, 1])


@pytest.mark.parametrize("backend", [SyncVectorEnv, AsyncVectorEnv])
def test_time_limit_validation(backend):
    """Reject missing/different limits, without losing async workers on missing limits."""
    with closing(vector_env(backend, [ClockEnv], AutoresetMode.NEXT_STEP)) as base:
        with pytest.raises(ValueError, match="max_episode_steps"):
            TimeAwareObservation(base)
        base.reset()
        base.step([0])
    with closing(
        vector_env(
            backend, [make_env, partial(make_env, limit=6)], AutoresetMode.NEXT_STEP
        )
    ) as base:
        with pytest.raises(ValueError, match="common episode time limit"):
            TimeAwareObservation(base)


def test_metadata_and_key_validation():
    """Match vector metadata conventions and scalar Dict collision validation."""
    with closing(SyncVectorEnv([partial(make_env, space_kind="dict")])) as base:
        with pytest.raises(ValueError, match="already exists"):
            TimeAwareObservation(base, dict_time_key="z")
        base.metadata = {"autoreset_mode": "NextStep"}
        with pytest.raises(TypeError, match="AutoresetMode"):
            TimeAwareObservation(base)
        base.metadata = {}
        with pytest.warns(UserWarning, match="missing.*autoreset_mode"):
            env = TimeAwareObservation(base)
        env.reset()
        env.step([0])
        np.testing.assert_array_equal(env.timesteps, [1])


def make_spec_env(limit=7):
    """Expose a spec limit without a TimeLimit wrapper, as the scalar API permits."""
    env = ClockEnv()
    env.spec = gym.envs.registration.EnvSpec("Clock-v0", max_episode_steps=limit)
    return env


@pytest.mark.parametrize("backend", [SyncVectorEnv, AsyncVectorEnv])
def test_spec_limit(backend):
    """Use a public spec even when no private TimeLimit attribute is available."""
    with closing(vector_env(backend, [make_spec_env], AutoresetMode.NEXT_STEP)) as base:
        env = TimeAwareObservation(base, flatten=False, normalize_time=True)
        env.reset()
        np.testing.assert_allclose(env.step([0])[0]["time"], [[1 / 7]])
        assert env.max_timesteps == 7
    with closing(
        vector_env(backend, [partial(make_spec_env, 0)], AutoresetMode.NEXT_STEP)
    ) as base:
        with pytest.raises(ValueError, match="positive integer"):
            TimeAwareObservation(base)


def test_scope_validation():
    """Reject unsupported native backends and different observation bounds explicitly."""
    native = gym.vector.VectorEnv()
    native.metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}
    with pytest.raises(TypeError, match="SyncVectorEnv or AsyncVectorEnv"):
        TimeAwareObservation(native)

    def different_space():
        """Keep the shape but change bounds to exercise observation_mode='different'."""
        env = make_env()
        env.observation_space = Box(0, 2, shape=(1,), dtype=np.float32)
        return env

    with closing(
        SyncVectorEnv([make_env, different_space], observation_mode="different")
    ) as base:
        with pytest.raises(ValueError, match="identical observation spaces"):
            TimeAwareObservation(base)
