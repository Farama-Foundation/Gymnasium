"""Tests that gym.make works as expected."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.wrappers import (
    AutoResetWrapper,
    HumanRendering,
    OrderEnforcing,
    TimeLimit,
)
from gymnasium.wrappers.env_checker import PassiveEnvChecker
from tests.envs.registration.utils_envs import ArgumentEnv
from tests.envs.utils import all_testing_env_specs
from tests.testing_env import GenericTestEnv, old_step_func
from tests.wrappers.utils import has_wrapper


try:
    import shimmy
except ImportError:
    shimmy = None


@pytest.fixture(scope="function")
def register_testing_envs():
    """Registers testing envs for `gym.make`"""
    gym.register(
        id="test.ArgumentEnv-v0",
        entry_point="tests.envs.registration.utils_envs:ArgumentEnv",
        kwargs={
            "arg1": "arg1",
            "arg2": "arg2",
        },
    )

    gym.register(
        id="test/NoHuman-v0",
        entry_point="tests.envs.registration.utils_envs:NoHuman",
    )
    gym.register(
        id="test/NoHumanOldAPI-v0",
        entry_point="tests.envs.registration.utils_envs:NoHumanOldAPI",
    )

    gym.register(
        id="test/NoHumanNoRGB-v0",
        entry_point="tests.envs.registration.utils_envs:NoHumanNoRGB",
    )

    gym.register(
        id="test/NoRenderModesMetadata-v0",
        entry_point="tests.envs.registration.utils_envs:NoRenderModesMetadata",
    )

    yield

    del gym.envs.registration.registry["test.ArgumentEnv-v0"]
    del gym.envs.registration.registry["test/NoRenderModesMetadata-v0"]
    del gym.envs.registration.registry["test/NoHuman-v0"]
    del gym.envs.registration.registry["test/NoHumanOldAPI-v0"]
    del gym.envs.registration.registry["test/NoHumanNoRGB-v0"]


def test_make():
    """Test basic `gym.make`."""
    env = gym.make("CartPole-v1")
    assert env.spec is not None
    assert env.spec.id == "CartPole-v1"
    assert isinstance(env.unwrapped, CartPoleEnv)
    env.close()


def test_make_deprecated():
    """Test make with a deprecated environment (i.e., doesn't exist)."""
    with warnings.catch_warnings(record=True):
        with pytest.raises(
            gym.error.Error,
            match=re.escape(
                "Environment version v0 for `Humanoid` is deprecated. Please use `Humanoid-v4` instead."
            ),
        ):
            gym.make("Humanoid-v0")


def test_make_max_episode_steps(register_testing_envs):
    # Default, uses the spec's
    env = gym.make("CartPole-v1")
    assert has_wrapper(env, TimeLimit)
    assert env.spec is not None
    assert env.spec.max_episode_steps == gym.spec("CartPole-v1").max_episode_steps
    env.close()

    # Custom max episode steps
    assert gym.spec("CartPole-v1").max_episode_steps != 100
    env = gym.make("CartPole-v1", max_episode_steps=100)
    assert has_wrapper(env, TimeLimit)
    assert env.spec is not None
    assert env.spec.max_episode_steps == 100
    env.close()

    # Env spec has no max episode steps
    assert gym.spec("test.ArgumentEnv-v0").max_episode_steps is None
    env = gym.make("test.ArgumentEnv-v0", arg1=None, arg2=None, arg3=None)
    assert env.spec is not None
    assert env.spec.max_episode_steps is None
    assert has_wrapper(env, TimeLimit) is False
    env.close()


def test_make_autoreset():
    """Tests that `gym.make` autoreset wrapper is applied only when `gym.make(..., autoreset=True)`."""
    env = gym.make("CartPole-v1")
    assert has_wrapper(env, AutoResetWrapper) is False
    env.close()

    env = gym.make("CartPole-v1", autoreset=False)
    assert has_wrapper(env, AutoResetWrapper) is False
    env.close()

    env = gym.make("CartPole-v1", autoreset=True)
    assert has_wrapper(env, AutoResetWrapper)
    env.close()


@pytest.mark.parametrize(
    "registration_disabled, make_disabled, if_disabled",
    [
        [False, False, False],
        [False, True, True],
        [True, False, False],
        [True, True, True],
        [False, None, False],
        [True, None, True],
    ],
)
def test_make_disable_env_checker(
    registration_disabled: bool, make_disabled: bool | None, if_disabled: bool
):
    """Tests that `gym.make` disable env checker is applied only when `gym.make(..., disable_env_checker=False)`.

    The ordering is 1. if the `make(..., disable_env_checker=...)` is bool, then the `registration(..., disable_env_checker=...)`
    """
    gym.register(
        "testing-env-v0",
        lambda: GenericTestEnv(),
        disable_env_checker=registration_disabled,
    )

    # Test when the registered EnvSpec.disable_env_checker = False
    env = gym.make("testing-env-v0", disable_env_checker=make_disabled)
    assert has_wrapper(env, PassiveEnvChecker) is not if_disabled
    env.close()

    del gym.registry["testing-env-v0"]


def test_make_apply_api_compatibility():
    """Test the API compatibility wrapper."""
    gym.register(
        "testing-old-env",
        lambda: GenericTestEnv(step_func=old_step_func),
        apply_api_compatibility=True,
        max_episode_steps=3,
    )
    # Apply the environment compatibility and check it works as intended
    env = gym.make("testing-old-env")
    assert isinstance(env.unwrapped, gym.wrappers.EnvCompatibility)

    env.reset()
    assert len(env.step(env.action_space.sample())) == 5
    env.step(env.action_space.sample())
    _, _, termination, truncation, _ = env.step(env.action_space.sample())
    assert termination is False and truncation is True

    # Turn off the spec api compatibility
    gym.spec("testing-old-env").apply_api_compatibility = False
    env = gym.make("testing-old-env")
    assert isinstance(env.unwrapped, gym.wrappers.EnvCompatibility) is False
    env.reset()
    with pytest.raises(
        ValueError, match=re.escape("not enough values to unpack (expected 5, got 4)")
    ):
        env.step(env.action_space.sample())

    # Apply the environment compatibility and check it works as intended
    env = gym.make("testing-old-env", apply_api_compatibility=True)
    assert isinstance(env.unwrapped, gym.wrappers.EnvCompatibility)

    env.reset()
    assert len(env.step(env.action_space.sample())) == 5
    env.step(env.action_space.sample())
    _, _, termination, truncation, _ = env.step(env.action_space.sample())
    assert termination is False and truncation is True

    del gym.registry["testing-old-env"]


def test_make_order_enforcing():
    """Checks that gym.make wrappers the environment with the OrderEnforcing wrapper."""
    assert all(spec.order_enforce is True for spec in all_testing_env_specs)

    env = gym.make("CartPole-v1")
    assert has_wrapper(env, OrderEnforcing)
    # We can assume that there all other specs will also have the order enforcing
    env.close()

    gym.register(
        id="test.OrderlessArgumentEnv-v0",
        entry_point="tests.envs.registration.utils_envs:ArgumentEnv",
        order_enforce=False,
        kwargs={"arg1": None, "arg2": None, "arg3": None},
    )

    env = gym.make("test.OrderlessArgumentEnv-v0")
    assert has_wrapper(env, OrderEnforcing) is False
    env.close()

    # There is no `make(..., order_enforcing=...)` so we don't test that


def test_make_render_mode():
    """Test the `make(..., render_mode=...)`, in particular, if to apply the `RenderCollection` or the `HumanRendering`."""
    env = gym.make("CartPole-v1", render_mode=None)
    assert env.render_mode is None
    env.close()

    assert "rgb_array" in env.metadata["render_modes"]
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    assert env.render_mode == "rgb_array"
    env.close()

    assert "no-render-mode" not in env.metadata["render_modes"]
    # cartpole is special that it doesn't check the render_mode passed at initialisation
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "\x1b[33mWARN: The environment is being initialised with render_mode='no-render-mode' that is not in the possible render_modes (['human', 'rgb_array']).\x1b[0m"
        ),
    ):
        env = gym.make("CartPole-v1", render_mode="no-render-mode")
        assert env.render_mode == "no-render-mode"
        env.close()


def test_make_render_collection():
    # Make sure that render_mode is applied correctly
    env = gym.make("CartPole-v1", render_mode="rgb_array_list")
    assert has_wrapper(env, gym.wrappers.RenderCollection)
    assert env.render_mode == "rgb_array_list"
    assert env.unwrapped.render_mode == "rgb_array"

    env.reset()
    renders = env.render()
    assert isinstance(
        renders, list
    )  # Make sure that the `render` method does what is supposed to
    assert isinstance(renders[0], np.ndarray)
    env.close()


def test_make_human_rendering(register_testing_envs):
    # Make sure that native rendering is used when possible
    env = gym.make("CartPole-v1", render_mode="human")
    assert not has_wrapper(env, HumanRendering)  # Should use native human-rendering
    assert env.render_mode == "human"
    env.close()

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "You are trying to use 'human' rendering for an environment that doesn't natively support it. The HumanRendering wrapper is being applied to your environment."
        ),
    ):
        # Make sure that `HumanRendering` is applied here as the environment doesn't use native rendering
        env = gym.make("test/NoHuman-v0", render_mode="human")
        assert has_wrapper(env, HumanRendering)
        assert env.render_mode == "human"
        env.close()

    with pytest.raises(
        TypeError, match=re.escape("got an unexpected keyword argument 'render_mode'")
    ):
        gym.make(
            "test/NoHumanOldAPI-v0",
            render_mode="rgb_array_list",
        )

    # Make sure that an additional error is thrown a user tries to use the wrapper on an environment with old API
    with warnings.catch_warnings(record=True):
        with pytest.raises(
            gym.error.Error,
            match=re.escape(
                "You passed render_mode='human' although test/NoHumanOldAPI-v0 doesn't implement human-rendering natively."
            ),
        ):
            gym.make("test/NoHumanOldAPI-v0", render_mode="human")

    # This test ensures that the additional exception "Gym tried to apply the HumanRendering wrapper but it looks like
    # your environment is using the old rendering API" is *not* triggered by a TypeError that originate from
    # a keyword that is not `render_mode`
    with pytest.raises(
        TypeError,
        match=re.escape("got an unexpected keyword argument 'render'"),
    ):
        gym.make("CarRacing-v2", render="human")

    # This test checks that a user can create an environment without the metadata including the render mode
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "\x1b[33mWARN: The environment is being initialised with render_mode='rgb_array' that is not in the possible render_modes ([]).\x1b[0m"
        ),
    ):
        gym.make("test/NoRenderModesMetadata-v0", render_mode="rgb_array")


def test_make_kwargs(register_testing_envs):
    env = gym.make(
        "test.ArgumentEnv-v0",
        arg2="override_arg2",
        arg3="override_arg3",
    )
    assert env.spec is not None
    assert env.spec.id == "test.ArgumentEnv-v0"
    assert env.spec.kwargs == {
        "arg1": "arg1",
        "arg2": "override_arg2",
        "arg3": "override_arg3",
    }

    assert isinstance(env.unwrapped, ArgumentEnv)
    assert env.arg1 == "arg1"
    assert env.arg2 == "override_arg2"
    assert env.arg3 == "override_arg3"
    env.close()


def test_import_module_during_make():
    # Test custom environment which is registered at make
    assert "RegisterDuringMake-v0" not in gym.registry
    env = gym.make(
        "tests.envs.registration.utils_unregistered_env:RegisterDuringMake-v0"
    )
    assert "RegisterDuringMake-v0" in gym.registry
    from tests.envs.registration.utils_unregistered_env import RegisterDuringMakeEnv

    assert isinstance(env.unwrapped, RegisterDuringMakeEnv)
    env.close()

    del gym.registry["RegisterDuringMake-v0"]
