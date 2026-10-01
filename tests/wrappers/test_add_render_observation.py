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


def test_record_constructor_args_roundtrip():
    """Saved constructor args must match the wrapper's own parameter names.

    ``RecordConstructorArgs`` stores these kwargs into the env spec, and
    reconstructing a wrapped env splats them back into ``__init__``. If the
    saved names differ from the parameters, that reconstruction raises
    ``TypeError``.
    """

    def make_env():
        return GenericTestEnv(
            observation_space=spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32),
            render_mode="rgb_array",
            render_func=image_render_func,
        )

    wrapped_env = AddRenderObservation(make_env(), render_only=True)

    saved_kwargs = wrapped_env._saved_kwargs
    assert set(saved_kwargs) == {"render_only", "render_key", "obs_key"}

    # Reconstruction (as performed from the env spec) must not raise.
    AddRenderObservation(make_env(), **saved_kwargs)


@pytest.mark.parametrize("key", ("state", "pixels"))
def test_single_array_observation_rejects_conflicting_keys(key):
    """Conflicting keys must not silently replace the original observation."""
    env = GenericTestEnv(
        render_mode="rgb_array",
        render_func=image_render_func,
    )

    with pytest.raises(ValueError, match="render_key and obs_key must be different"):
        AddRenderObservation(env, render_only=False, render_key=key, obs_key=key)


@pytest.mark.parametrize("dict_observation", (False, True))
def test_equal_keys_when_obs_key_is_unused(dict_observation):
    """Equal keys are harmless when the wrapper does not use obs_key."""
    state_space = spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float32)
    env = GenericTestEnv(
        observation_space=(
            spaces.Dict(state=state_space) if dict_observation else state_space
        ),
        render_mode="rgb_array",
        render_func=image_render_func,
    )
    wrapped_env = AddRenderObservation(
        env,
        render_only=not dict_observation,
        render_key="pixels",
        obs_key="pixels",
    )

    observations = (wrapped_env.reset()[0], wrapped_env.step(None)[0])
    for obs in observations:
        assert obs in wrapped_env.observation_space
        if dict_observation:
            assert set(obs) == {"state", "pixels"}
            assert obs["state"] in state_space
            np.testing.assert_array_equal(obs["pixels"], env.render())
        else:
            np.testing.assert_array_equal(obs, env.render())
