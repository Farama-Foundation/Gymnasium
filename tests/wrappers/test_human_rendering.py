"""Test suite of HumanRendering wrapper."""
import re
from typing import Dict

import pytest

import gymnasium as gym
from gymnasium.wrappers import HumanRendering, HumanVectorRendering


@pytest.mark.parametrize(
    "mode, apply_wrapper, wrap_exception, exception_message",
    [
        [
            None,
            True,
            AssertionError,
            "Render mode was not set for <TimeLimit<OrderEnforcing<CartPoleEnv<CartPole-v1>>>>"
            ", set to acceptable 'rgb_array'",
        ],
        ["rgb_array", True, None, None],
        ["rgb_array_list", True, None, None],
        ["human", False, None, None],  # Automatic applied
        [
            "human",
            True,
            AssertionError,  # The HumanRenderer has already been applied automatically
            "Expected env.render_mode to be one of 'rgb_array' or 'rgb_array_list' but got 'human'",
        ],
        [
            "whatever",
            True,
            AssertionError,
            "Expected env.render_mode to be one of 'rgb_array' or 'rgb_array_list' but got 'whatever'",
        ],
    ],
)
def test_human_rendering(
    mode: str,
    apply_wrapper: bool,
    wrap_exception: Exception,
    exception_message: str,
):
    env = gym.make("CartPole-v1", render_mode=mode, disable_env_checker=True)

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

    assert env.render_mode == "human"
    env.reset()

    for _ in range(75):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()

    env.close()


@pytest.mark.parametrize(
    "mode, apply_wrapper, expected_exception, exception_message",
    [
        [
            None,
            True,
            {"wrap": AssertionError},
            "Render mode was not set for CartPoleVectorEnv(CartPole-v1, num_envs=3), set to acceptable 'rgb_array'",
        ],
        ["rgb_array", True, None, None],
        [
            "rgb_array_list",  # interpreted as rgb_array
            True,
            {"reset": AssertionError},  # There is no vectorized RenderCollection
            "Expected render to return a list of (list of environment) RGB arrays",
        ],
        ["human", False, None, None],
        [
            "human",
            True,
            {
                "wrap": AssertionError
            },  # No automatic wrapping for vectorized environments (yet)
            "Expected env.render_mode to be one of 'rgb_array' or 'rgb_array_list' but got 'human'",
        ],
        [
            "whatever",
            True,
            {"wrap": AssertionError},
            "Expected env.render_mode to be one of 'rgb_array' or 'rgb_array_list' but got 'whatever'",
        ],
    ],
)
def test_human_vector_rendering(
    mode: str,
    apply_wrapper: bool,
    expected_exception: Dict[str, Exception],
    exception_message: str,
):
    exception_message = "" if exception_message is None else exception_message
    num_envs = 3

    def expect_exception_in(named):
        return None if expected_exception is None else expected_exception.get(named)

    make_exception = expect_exception_in("make")
    wrap_exception = expect_exception_in("wrap")
    reset_exception = expect_exception_in("reset")

    if make_exception:
        with pytest.raises(
            make_exception,
            match=re.escape(
                exception_message,
            ),
        ):
            env = gym.make_vec(
                "CartPole-v1", num_envs=num_envs, vector_kwargs={"render_mode": mode}
            )
        return
    else:
        env = gym.make_vec(
            "CartPole-v1", num_envs=num_envs, vector_kwargs={"render_mode": mode}
        )

    if apply_wrapper:
        if wrap_exception:
            with pytest.raises(
                wrap_exception,
                match=re.escape(
                    exception_message,
                ),
            ):
                env = HumanVectorRendering(env)
            env.close()
            return
        else:
            env = HumanVectorRendering(env)

        assert env.render_mode == "human"

    if reset_exception:
        with pytest.raises(
            reset_exception,
            match=re.escape(
                exception_message,
            ),
        ):
            env.reset()
        env.close()
        return
    else:
        env.reset()

    assert (
        env.render_mode == "human"
    )  # Reaching this point we should have rendered something, check if mode is human

    for _ in range(75):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated.all() or truncated.all():
            env.reset()

    env.close()
