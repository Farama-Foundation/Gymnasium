"""Tests for the pixel observation wrapper."""
from typing import Optional

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RenderObservationV0


STATE_KEY = "state"


class FakeEnvironment(gym.Env):
    def __init__(self, render_mode="rgb_array"):
        self.action_space = spaces.Box(shape=(1,), low=-1, high=1, dtype=np.float32)
        self.render_mode = render_mode

    def render(self, mode="human", width=32, height=32):
        image_shape = (height, width, 3)
        return np.zeros(image_shape, dtype=np.uint8)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        observation = self.observation_space.sample()
        return observation, {}

    def step(self, action):
        del action
        observation = self.observation_space.sample()
        reward, terminal, info = 0.0, False, {}
        return observation, reward, terminal, info


class FakeArrayObservationEnvironment(FakeEnvironment):
    def __init__(self, *args, **kwargs):
        self.observation_space = spaces.Box(
            shape=(2,), low=-1, high=1, dtype=np.float32
        )
        super().__init__(*args, **kwargs)


class FakeDictObservationEnvironment(FakeEnvironment):
    def __init__(self, *args, **kwargs):
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32),
            }
        )
        super().__init__(*args, **kwargs)


@pytest.mark.parametrize("pixels_only", (True, False))
def test_dict_observation(pixels_only):
    pixel_key = "rgb"

    env = FakeDictObservationEnvironment()

    # Make sure we are testing the right environment for the test.
    assert isinstance(env.observation_space, spaces.Dict)

    # width, height = (320, 240)

    # The wrapper should only add one observation.
    wrapped_env = RenderObservationV0(
        env,
        render_key=pixel_key,
        render_only=pixels_only,
        # render_kwargs={pixel_key: {"width": width, "height": height}},
    )
    obs, info = wrapped_env.reset()
    if pixels_only:
        assert isinstance(wrapped_env.observation_space, spaces.Box)
        assert isinstance(obs, np.ndarray)

        rendered_obs = obs
    else:
        assert isinstance(wrapped_env.observation_space, spaces.Dict)

        expected_keys = [pixel_key] + list(env.observation_space.spaces.keys())
        assert list(wrapped_env.observation_space.spaces.keys()) == expected_keys

        assert isinstance(obs, dict)
        rendered_obs = obs[pixel_key]

    # Check that the added space item is consistent with the added observation.
    # assert rendered_obs.shape == (height, width, 3)
    assert rendered_obs.ndim == 3
    assert rendered_obs.dtype == np.uint8


@pytest.mark.parametrize("pixels_only", (True, False))
def test_single_array_observation(pixels_only):
    pixel_key = "depth"

    env = FakeArrayObservationEnvironment()
    assert isinstance(env.observation_space, spaces.Box)

    # The wrapper should only add one observation.
    wrapped_env = RenderObservationV0(
        env,
        render_key=pixel_key,
        render_only=pixels_only,
        # render_kwargs={pixel_key: {"width": width, "height": height}},
    )
    obs, info = wrapped_env.reset()
    if pixels_only:
        assert isinstance(wrapped_env.observation_space, spaces.Box)
        assert isinstance(obs, np.ndarray)

        rendered_obs = obs
    else:
        assert isinstance(wrapped_env.observation_space, spaces.Dict)

        expected_keys = [pixel_key, "state"]
        assert list(wrapped_env.observation_space.spaces.keys()) == expected_keys

        assert isinstance(obs, dict)
        rendered_obs = obs[pixel_key]

    # Check that the added space item is consistent with the added observation.
    # assert rendered_obs.shape == (height, width, 3)
    assert rendered_obs.ndim == 3
    assert rendered_obs.dtype == np.uint8
