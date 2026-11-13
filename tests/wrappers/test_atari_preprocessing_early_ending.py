"""Check Atari screen pooling independently of ROMs and episode lengths."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

pytest.importorskip("cv2")


class ScreenEnv(gym.Env):
    """Expose deterministic RGB and grayscale screens through the ALE API."""

    def __init__(self, end_step, ending):
        """Choose when the episode terminates, truncates, or loses a life."""
        self.observation_space = gym.spaces.Box(0, 255, (2, 2, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self._frameskip = 1
        self.ale = self
        self.end_step = end_step
        self.ending = ending
        self.frames = np.array(
            [
                [[250, 250], [250, 250]],
                [[10, 90], [30, 70]],
                [[80, 20], [60, 40]],
                [[30, 70], [10, 90]],
                [[60, 40], [80, 20]],
                [[20, 60], [40, 80]],
                [[50, 10], [70, 30]],
            ],
            dtype=np.uint8,
        )

    def reset(self, *, seed=None, options=None):
        """Start from a bright screen distinct from all step screens."""
        super().reset(seed=seed)
        self.steps = 0
        return self._rgb(), {"reset": True}

    def _rgb(self):
        return np.stack(
            [
                self.frames[self.steps],
                self.frames[self.steps] + 1,
                self.frames[self.steps] + 2,
            ],
            axis=-1,
        )

    def step(self, action):
        """Advance one screen and reject steps past the chosen ending."""
        assert action == 1
        assert self.steps < self.end_step
        self.steps += 1
        ended = self.steps == self.end_step
        return (
            self._rgb(),
            float(self.steps),
            ended and self.ending == "terminated",
            ended and self.ending == "truncated",
            {"step": self.steps},
        )

    def lives(self):
        """Drop a life only at the configured life-loss ending."""
        return 2 if self.ending == "life_loss" and self.steps == self.end_step else 3

    def getScreenGrayscale(self, output):
        """Write the current grayscale screen into the supplied buffer."""
        output[:] = self.frames[self.steps]

    def getScreenRGB(self, output):
        """Write the current RGB screen into the supplied buffer."""
        output[:] = self._rgb()


def expected_screen(base, first, last, grayscale, scaled):
    """Pool the actual screens from the last two steps, without resizing."""
    expected = base.frames[first : last + 1].max(axis=0)
    if not grayscale:
        expected = np.stack([expected, expected + 1, expected + 2], axis=-1)
    if scaled:
        expected = expected.astype(np.float32) / 255.0
    return expected


@pytest.mark.parametrize("grayscale", [True, False])
@pytest.mark.parametrize("scaled", [True, False])
@pytest.mark.parametrize("ending", ["terminated", "truncated", "life_loss"])
@pytest.mark.parametrize(
    "skip,end_step",
    [(skip, end) for skip in (1, 2, 4) for end in range(1, skip + 1)],
)
def test_ending_screen(skip, end_step, ending, grayscale, scaled):
    """An ending at any skipped step must include its final screen."""
    base = ScreenEnv(end_step, ending)
    env = AtariPreprocessing(
        base,
        frame_skip=skip,
        noop_max=0,
        screen_size=2,
        grayscale_obs=grayscale,
        scale_obs=scaled,
        terminal_on_life_loss=ending == "life_loss",
    )
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(1)

    expected = expected_screen(base, max(1, end_step - 1), end_step, grayscale, scaled)
    np.testing.assert_array_equal(obs, expected)
    assert env.observation_space.contains(obs)
    assert reward == sum(range(1, end_step + 1))
    assert terminated is (ending != "truncated")
    assert truncated is (ending == "truncated")
    assert info == {"step": end_step}
    assert base.steps == end_step
    env.close()


@pytest.mark.parametrize("grayscale", [True, False])
@pytest.mark.parametrize("ending", ["terminated", "truncated", "life_loss"])
@pytest.mark.parametrize("reset_before_ending", [True, False])
def test_ending_discards_previous_screens(grayscale, ending, reset_before_ending):
    """An early ending must not mix in a previous call's or episode's screens."""
    base = ScreenEnv(6, ending)
    env = AtariPreprocessing(
        base,
        frame_skip=4,
        noop_max=0,
        screen_size=2,
        grayscale_obs=grayscale,
        terminal_on_life_loss=ending == "life_loss",
    )
    env.reset(seed=0)
    previous_obs, reward, terminated, truncated, info = env.step(1)
    previous_expected = expected_screen(base, 3, 4, grayscale, False)
    np.testing.assert_array_equal(previous_obs, previous_expected)
    assert reward == 10 and not terminated and not truncated
    assert info == {"step": 4}

    if reset_before_ending:
        base.end_step = 1
        env.reset()
        first, last, expected_reward = 1, 1, 1
    else:
        first, last, expected_reward = 5, 6, 11

    obs, reward, terminated, truncated, info = env.step(1)
    np.testing.assert_array_equal(
        obs, expected_screen(base, first, last, grayscale, False)
    )
    np.testing.assert_array_equal(previous_obs, previous_expected)
    assert reward == expected_reward
    assert terminated is (ending != "truncated")
    assert truncated is (ending == "truncated")
    assert info == {"step": last}
    env.close()
