"""Test suite for RenderObservation wrapper."""

import numpy as np
import pytest

from gymnasium import spaces
from gymnasium.wrappers import AddRenderObservation
from tests.testing_env import GenericTestEnv


STATE_KEY = "state"


def image_render_func(self):
    return np.zeros((32, 32, 3), dtype=np.uint8)


@pytest.mark.parametrize("pixels_only", (True, False))
def test_dict_observation(pixels_only, pixel_key="rgb"):
    env = GenericTestEnv(
        observation_space=spaces.Dict(
            state=spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32)
        ),
        render_mode="rgb_array",
        render_func=image_render_func,
    )

    # Make sure we are testing the right environment for the test.
    assert isinstance(env.observation_space, spaces.Dict)

    # width, height = (320, 240)

    # The wrapper should only add one observation.
    wrapped_env = AddRenderObservation(
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

    env = GenericTestEnv(
        observation_space=spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32),
        render_mode="rgb_array",
        render_func=image_render_func,
    )
    assert isinstance(env.observation_space, spaces.Box)

    # The wrapper should only add one observation.
    wrapped_env = AddRenderObservation(
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
