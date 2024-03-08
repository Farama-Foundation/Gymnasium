"""Test suite of HumanRendering wrapper."""
from __future__ import annotations

import pytest

import gymnasium as gym
from gymnasium.wrappers import HumanRendering


@pytest.mark.parametrize(
    "mode, apply_wrapper, reset_exception, exception_message",
    [
        [
            None,
            True,
            AssertionError,
            r"Render mode was not set for [^>]+CartPole[^,]+, set to acceptable 'rgb_array'",
        ],
        ["rgb_array", True, None, None],
        ["rgb_array_list", True, None, None],
        ["human", False, None, None],  # Automatic applied
        [
            "human",
            True,
            AssertionError,  # The HumanRenderer has already been applied automatically
            r"Expected [^>]+CartPole[^ ]+ 'render_mode' to be one of 'rgb_array' or 'rgb_array_list' but got 'human'",
        ],
        [
            "whatever",
            True,
            AssertionError,
            r"Expected [^>]+CartPole[^ ]+ 'render_mode' to be one of 'rgb_array'"
            r" or 'rgb_array_list' but got 'whatever'",
        ],
        [
            "whatever",
            True,
            AssertionError,
            r"Expected [^>]+CartPole[^ ]+ 'render_mode' to be one of 'rgb_array'"
            r" or 'rgb_array_list' but got 'whatever'",
        ],
    ],
)
def test_human_rendering(
    mode: str,
    apply_wrapper: bool,
    reset_exception: type[Exception],
    exception_message: str,
):
    env = gym.make("CartPole-v1", render_mode=mode, disable_env_checker=True)

    if apply_wrapper:
        env = HumanRendering(env)
        assert (
            env.render_mode == "human"
            if mode in ["rgb_array", "rgb_array_list"]
            else env.render_mode == mode
        ), f"Unexpected render mode {env.render_mode}"

    if reset_exception:
        with pytest.raises(
            reset_exception,
            match=exception_message,
        ):
            env.reset()
        return
    else:
        env.reset()

    assert (
        env.render_mode == "human"
    ), "Can't verify actual rendering, verifying render mode"

    for _ in range(75):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()

    env.close()
