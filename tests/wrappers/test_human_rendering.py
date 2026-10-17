"""Test suite of HumanRendering wrapper."""

import re
from copy import deepcopy

import numpy as np
import pygame
import pytest

import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation, HumanRendering, RenderCollection


def test_human_rendering():
    for mode in ["rgb_array", "rgb_array_list"]:
        env = HumanRendering(
            gym.make("CartPole-v1", render_mode=mode, disable_env_checker=True)
        )
        assert env.render_mode == "human"
        env.reset()

        for _ in range(75):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()

        env.close()

    env = gym.make("CartPole-v1", render_mode="human")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected env.render_mode to be one of ['rgb_array', 'rgb_array_list', 'depth_array', 'depth_array_list'] but got 'human'"
        ),
    ):
        HumanRendering(env)
    env.close()


class CountingRGBEnv(gym.Env):
    """Render a reused nonsquare image and expose lifecycle calls for cache tests."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1000}

    def __init__(self, render_mode="rgb_array"):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(100)
        self.render_mode = render_mode
        self.frame = np.zeros((16, 24, 3), dtype=np.uint8)
        self.render_calls = self.close_calls = self.steps = 0
        self.failure = None
        self.copy_frames = False

    def reset(self, *, seed=None, options=None):
        """Restart the image sequence, optionally failing before rendering."""
        super().reset(seed=seed)
        if self.failure == "reset":
            raise RuntimeError("reset failed")
        self.steps = 0
        return self.steps, {"seed": seed, "options": options}

    def step(self, action):
        """Advance the image and terminate after three steps."""
        self.steps += 1
        if self.failure == "step":
            raise RuntimeError("step failed")
        return self.steps, 1.0, self.steps == 3, False, {}

    def render(self):
        """Return the same mutable buffer with spatially distinct RGB values."""
        self.render_calls += 1
        if self.failure == "render":
            raise RuntimeError("render failed")
        if (
            isinstance(self.frame, np.ndarray)
            and self.frame.ndim == 3
            and self.frame.shape[-1] == 3
        ):
            self.frame[..., 0] = 20 + self.steps * 20
            self.frame[..., 1] = np.arange(self.frame.shape[1])
            self.frame[..., 2] = np.arange(self.frame.shape[0])[:, None]
        return self.frame.copy() if self.copy_frames else self.frame

    def close(self):
        """Count underlying close calls."""
        self.close_calls += 1


def test_rgb_cache_pixels_ownership_and_call_counts(monkeypatch):
    """Cache exactly the displayed HWC pixels without repeating render or event calls."""
    base = CountingRGBEnv()
    metadata = deepcopy(base.metadata)
    env = HumanRendering(base, return_rgb_array=True)
    counts = {"pump": 0, "flip": 0, "tick": 0}
    for name, module in [("pump", pygame.event), ("flip", pygame.display)]:
        original = getattr(module, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    class Clock:
        def tick(self, fps):
            counts["tick"] += 1

    monkeypatch.setattr(pygame.time, "Clock", Clock)
    try:
        assert env.render_mode == "rgb_array"
        assert env.metadata["render_modes"] == ["rgb_array"]
        assert base.metadata == metadata and env.metadata is not base.metadata
        with pytest.raises(gym.error.ResetNeeded):
            env.render()
        with pytest.raises(gym.error.ResetNeeded):
            env.step(0)
        _, info = env.reset(seed=7, options={"test": True})
        assert info == {"seed": 7, "options": {"test": True}}
        initial = env.render()
        displayed = pygame.surfarray.array3d(env.window).transpose(1, 0, 2)
        np.testing.assert_array_equal(initial, displayed)
        assert initial.shape == (16, 24, 3)
        base.frame[:] = 255
        np.testing.assert_array_equal(env.render(), initial)
        altered = env.render()
        altered[:] = 0
        np.testing.assert_array_equal(env.render(), initial)
        assert base.render_calls == 1 and counts == {"pump": 1, "flip": 1, "tick": 1}
        env.step(0)
        assert base.render_calls == 2
        assert np.all(env.render()[..., 0] == 40)
        assert np.all(initial[..., 0] == 20)
        env.step(0)
        _, _, terminal, _, _ = env.step(0)
        assert terminal and np.all(env.render()[..., 0] == 80)
        assert counts == {"pump": 4, "flip": 4, "tick": 4}
    finally:
        env.close()
    env.close()
    assert base.close_calls == 1
    assert env.window is env.clock is env.screen_size is None
    for call in [env.render, env.reset, lambda: env.step(0)]:
        with pytest.raises(gym.error.ClosedEnvironmentError):
            call()


@pytest.mark.parametrize("stage", ["reset", "step", "render", "display"])
def test_rgb_failed_updates_invalidate_cache(stage, monkeypatch):
    base = CountingRGBEnv()
    env = HumanRendering(base, return_rgb_array=True)
    try:
        env.reset()
        retained = env.render()
        with monkeypatch.context() as patch:
            if stage == "display":

                def fail():
                    raise RuntimeError("display failed")

                patch.setattr(pygame.display, "flip", fail)
            else:
                base.failure = stage
            with pytest.raises(RuntimeError, match="failed"):
                env.reset() if stage == "reset" else env.step(0)
        with pytest.raises(gym.error.ResetNeeded):
            env.render()
        with pytest.raises(gym.error.ResetNeeded):
            env.step(0)
        assert np.all(retained[..., 0] == 20)
        base.failure = None
        env.reset()
        assert env.render().shape == (16, 24, 3)
    finally:
        env.close()


@pytest.mark.parametrize(
    "frame",
    [
        None,
        [],
        np.zeros((16, 24), np.uint8),
        np.zeros((16, 24, 4), np.uint8),
        np.zeros((0, 24, 3), np.uint8),
        np.zeros((16, 24, 3), np.float32),
        np.zeros((20, 24, 3), np.uint8),
    ],
)
def test_rgb_invalid_frame_or_resize_requires_reset(frame):
    base = CountingRGBEnv()
    env = HumanRendering(base, return_rgb_array=True)
    try:
        env.reset()
        original = base.frame
        base.frame = frame
        with pytest.raises(ValueError):
            env.step(0)
        with pytest.raises(gym.error.ResetNeeded):
            env.render()
        base.frame = original
        env.reset()
        assert env.render().dtype == np.uint8
    finally:
        env.close()


@pytest.mark.parametrize(
    "mode", [None, "human", "rgb_array_list", "depth_array", "depth_array_list"]
)
def test_rgb_rejects_unsupported_modes(mode):
    base = CountingRGBEnv()
    base.render_mode = mode
    with pytest.raises(ValueError, match="requires"):
        HumanRendering(base, return_rgb_array=True)


def test_rgb_render_collection_and_observation_composition():
    base = CountingRGBEnv()
    env = RenderCollection(HumanRendering(base, return_rgb_array=True))
    try:
        env.reset()
        env.step(0)
        frames = env.render()
        assert len(frames) == 2 and env.render() == []
        assert base.render_calls == 2
    finally:
        env.close()
    base = CountingRGBEnv()
    with pytest.raises(ValueError, match="requires"):
        HumanRendering(RenderCollection(base), return_rgb_array=True)
    base = CountingRGBEnv()
    env = AddRenderObservation(HumanRendering(base, return_rgb_array=True))
    try:
        # AddRenderObservation resets at construction to infer its image space.
        before = base.render_calls
        observation, _ = env.reset()
        assert base.render_calls == before + 1
        assert observation in env.observation_space
    finally:
        env.close()


def test_default_human_rendering_does_not_cache_frames():
    base = CountingRGBEnv()
    env = HumanRendering(base)
    try:
        assert env.render_mode == "human" and env.render() is None
        env.reset()
        assert env.render() is None and env._cached_rgb_array is None
    finally:
        env.close()


def test_make_human_fallback_still_returns_none():
    """Automatic human mode does not silently opt into the new RGB return contract."""
    env_id = "test/CountingRGBFallback-v0"
    gym.register(env_id, entry_point=CountingRGBEnv)
    try:
        with pytest.warns(UserWarning, match="HumanRendering wrapper is being applied"):
            env = gym.make(env_id, render_mode="human")
        try:
            assert isinstance(env, HumanRendering)
            assert env.render_mode == "human" and not env.return_rgb_array
            env.reset()
            assert env.render() is None
        finally:
            env.close()
    finally:
        del gym.envs.registry[env_id]


def test_rgb_spec_roundtrip():
    """Reconstruction retains the explicitly enabled rendering contract."""
    env = HumanRendering(
        gym.make("CartPole-v1", render_mode="rgb_array"), return_rgb_array=True
    )
    try:
        reconstructed = gym.make(env.spec)
        try:
            assert reconstructed.render_mode == "rgb_array"
            assert reconstructed._saved_kwargs == {"return_rgb_array": True}
        finally:
            reconstructed.close()
    finally:
        env.close()
