"""Test suite of vector HumanRendering wrapper."""
from __future__ import annotations

import re

import pytest

import gymnasium as gym
from gymnasium.wrappers.vector import HumanRendering


@pytest.mark.parametrize(
    "mode, apply_wrapper, expected_exception, exception_message",
    [
        [
            None,
            True,
            {"reset": AssertionError},
            r"Render mode was not set for CartPoleVectorEnv[^\)]+\), set to acceptable 'rgb_array'",
        ],
        ["rgb_array", True, None, None],
        [
            "rgb_array_list",  # interpreted as rgb_array
            True,
            {"reset": AssertionError},  # There is no vectorized RenderCollection
            r"Expected render to return a list of \(list of environment\) RGB arrays",
        ],
        ["human", False, None, None],
        [
            "human",
            True,
            {
                "reset": AssertionError
            },  # No automatic wrapping for vectorized environments (yet)
            r"Expected CartPoleVectorEnv[^\)]+\) 'render_mode' to be one of 'rgb_array' or 'rgb_array_list' but got 'human'",
        ],
        [
            "whatever",
            True,
            {"reset": AssertionError},
            r"Expected CartPoleVectorEnv[^\)]+\) 'render_mode' to be one of 'rgb_array' or 'rgb_array_list' but got 'whatever'",
        ],
        [
            {},
            True,
            {"reset": AssertionError},
            r"Render mode was not set for CartPoleVectorEnv[^\)]+\), set to acceptable 'rgb_array'",
        ],
    ],
)
def test_human_vector_rendering(
    mode: str,
    apply_wrapper: bool,
    expected_exception: dict[str, type[Exception]],
    exception_message: str,
):
    exception_message = "" if exception_message is None else exception_message
    num_envs = 3

    def expect_exception_in(named):
        return None if expected_exception is None else expected_exception.get(named)

    make_exception = expect_exception_in("make")
    wrap_exception = expect_exception_in("wrap")
    reset_exception = expect_exception_in("reset")

    options = mode if isinstance(mode, dict) else {"render_mode": mode}
    mode = None if isinstance(mode, dict) else mode  # Mode to expect default if not set

    if make_exception:
        with pytest.raises(
            make_exception,
            match=re.escape(
                exception_message,
            ),
        ):
            gym.make_vec("CartPole-v1", num_envs=num_envs, vector_kwargs=options)
        return
    else:
        env = gym.make_vec("CartPole-v1", num_envs=num_envs, vector_kwargs=options)

    if apply_wrapper:
        if wrap_exception:
            with pytest.raises(
                wrap_exception,
                match=re.escape(
                    exception_message,
                ),
            ):
                env = HumanRendering(env)
            env.close()
            return
        else:
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
        env.close()
        return
    else:
        env.reset()

    assert (
        env.render_mode == "human"
    ), "Can't verify actual rendering, verifying render mode"

    for _ in range(75):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated.all() or truncated.all():
            env.reset()

    env.close()
