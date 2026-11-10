"""Test suite of AddWhiteNoise and ObstructView wrapper."""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.wrappers import AddWhiteNoise, HumanRendering, ObstructView


def test_white_noise_rendering():
    for mode in ["rgb_array"]:
        env = gym.make("CartPole-v1", render_mode=mode, disable_env_checker=True)
        env = AddWhiteNoise(env, probability_of_noise_per_pixel=0.5)
        env = HumanRendering(env)

        assert env.render_mode == "human"
        env.reset()

        for _ in range(75):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()

        env.close()

        env = gym.make("CartPole-v1", render_mode=mode, disable_env_checker=True)
        env = ObstructView(env, obstructed_pixels_ratio=0.5, obstruction_width=100)
        env = HumanRendering(env)

        assert env.render_mode == "human"
        env.reset()

        for _ in range(75):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()

        env.close()


@pytest.mark.parametrize("is_noise_grayscale", [False, True])
@pytest.mark.parametrize(
    "wrapper, kwargs",
    [
        (AddWhiteNoise, {"probability_of_noise_per_pixel": 0.5}),
        (ObstructView, {"obstructed_pixels_ratio": 0.5, "obstruction_width": 20}),
    ],
)
def test_noise_rendering_dtype(wrapper, kwargs, is_noise_grayscale):
    """Tests that the noisy frames keep the uint8 dtype of the environment's frames."""
    env = gym.make("CartPole-v1", render_mode="rgb_array", disable_env_checker=True)
    env = wrapper(env, is_noise_grayscale=is_noise_grayscale, **kwargs)
    env.reset(seed=123)

    frame = env.render()
    assert frame.dtype == np.uint8
    assert frame.shape == env.env.render().shape

    env.close()
