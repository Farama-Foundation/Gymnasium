"""Test suite of HumanRendering wrapper."""

import re

import pytest

import gymnasium as gym
from gymnasium.wrappers.vector import HumanRendering


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Ant-v4"])
@pytest.mark.parametrize("num_envs", [1, 3, 9])
@pytest.mark.parametrize("screen_size", [None, (400, 300), (300, 600), (600, 600)])
def test_num_envs_screen_size(env_id, num_envs, screen_size):
    envs = gym.make_vec(env_id, num_envs=num_envs, render_mode="rgb_array")
    envs = HumanRendering(envs, screen_size=screen_size)

    assert envs.render_mode == "human"

    envs.reset()
    for _ in range(25):
        envs.step(envs.action_space.sample())
    envs.close()


def test_render_modes():
    envs = HumanRendering(
        gym.make_vec("CartPole-v1", num_envs=3, render_mode="rgb_array_list")
    )
    assert envs.render_mode == "human"

    envs.reset()
    for _ in range(25):
        envs.step(envs.action_space.sample())
    envs.close()

    # HumanRenderer on human renderer should not work
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expected env.render_mode to be one of ['rgb_array', 'rgb_array_list', 'depth_array', 'depth_array_list'] but got 'human'"
        ),
    ):
        HumanRendering(envs)
