"""Test suite of HumanRendering wrapper."""

import re

import numpy as np
import pygame
import pytest

import gymnasium as gym
from gymnasium.wrappers.vector import HumanRendering
from tests.testing_env import GenericTestVectorEnv


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


@pytest.mark.parametrize("vectorization_mode", ["vector_entry_point", "sync", "async"])
def test_render_modes(vectorization_mode):
    envs = HumanRendering(
        gym.make_vec(
            "CartPole-v1",
            num_envs=3,
            render_mode="rgb_array_list",
            vectorization_mode=vectorization_mode,
        )
    )
    assert envs.render_mode == "human"

    envs.reset()
    for _ in range(25):
        envs.step(envs.action_space.sample())
    envs.close()

    # HumanRenderer on human renderer should not work
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected env.render_mode to be one of ['rgb_array', 'rgb_array_list', 'depth_array', 'depth_array_list'] but got 'human'"
        ),
    ):
        HumanRendering(envs)


def test_render_latest_frame_per_env():
    frames = [
        [np.full((2, 2, 3), value, dtype=np.uint8) for value in values]
        for values in ([10, 20], [30, 40, 50])
    ]
    envs = HumanRendering(
        GenericTestVectorEnv(
            num_envs=2,
            render_mode="rgb_array_list",
            metadata={"render_modes": ["rgb_array_list"], "render_fps": 0},
            render_func=lambda self: tuple(frames),
        ),
        screen_size=(4, 2),
    )

    try:
        envs.reset(seed=123)
        expected = np.concatenate([frames[0][-1], frames[1][-1]], axis=0)
        np.testing.assert_array_equal(pygame.surfarray.array3d(envs.window), expected)

        frames[0].append(np.full((2, 2, 3), 60, dtype=np.uint8))
        frames[1].append(np.full((2, 2, 3), 70, dtype=np.uint8))
        envs.step(envs.action_space.sample())
        expected = np.concatenate([frames[0][-1], frames[1][-1]], axis=0)
        np.testing.assert_array_equal(pygame.surfarray.array3d(envs.window), expected)
    finally:
        envs.close()
