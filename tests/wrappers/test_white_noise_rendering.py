"""Test suite of AddWhiteNoise and ObstructView wrapper."""

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
