"""Test suite of HumanRendering wrapper."""
import re

import pytest

import gymnasium as gym
from gymnasium.wrappers.vector import HumanRendering


def test_vector_human_rendering():
    for mode in ["rgb_array", "rgb_array_list"]:
        env = HumanRendering(gym.make_vec("CartPole-v1", num_envs=3, render_mode=mode))
        assert env.render_mode == "human"
        env.reset()

        for _ in range(75):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated.all() or truncated.all():
                env.reset()

        env.close()


def test_builtin_human_vector_rendering():
    """Note: vector classes should not try to implement human rendering, use wrapper!"""
    env = gym.make_vec(
        "CartPole-v1",
        num_envs=3,  # Test hint: do not use 4 as that is number of state parameters
        render_mode="human",
    )

    assert env.render_mode == "human", f"Unexpected render mode {env.render_mode}"

    env.reset()
    for _ in range(75):
        # Note: No error and nothing gets rendered on screen...
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated.all() or truncated.all():
            env.reset()

    # HumanRenderer on human renderer should not work
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expected env.render_mode to be one of ['rgb_array', 'rgb_array_list', 'depth_array', 'depth_array_list'] but got 'human'"
        ),
    ):
        HumanRendering(env)
    env.close()
