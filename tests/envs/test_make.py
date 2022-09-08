"""Tests that gymnasium.make works as expected."""

import re
import warnings
from copy import deepcopy

import numpy as np
import pytest

import gymnasium
from gymnasium.envs.classic_control import cartpole
from gymnasium.wrappers import AutoResetWrapper, HumanRendering, OrderEnforcing, TimeLimit
from gymnasium.wrappers.env_checker import PassiveEnvChecker
from tests.envs.test_envs import PASSIVE_CHECK_IGNORE_WARNING
from tests.envs.utils import all_testing_env_specs
from tests.envs.utils_envs import ArgumentEnv, RegisterDuringMakeEnv
from tests.testing_env import GenericTestEnv, old_step_fn
from tests.wrappers.utils import has_wrapper

gymnasium.register(
    "RegisterDuringMakeEnv-v0",
    entry_point="tests.envs.utils_envs:RegisterDuringMakeEnv",
)

gymnasium.register(
    id="test.ArgumentEnv-v0",
    entry_point="tests.envs.utils_envs:ArgumentEnv",
    kwargs={
        "arg1": "arg1",
        "arg2": "arg2",
    },
)

gymnasium.register(
    id="test/NoHuman-v0",
    entry_point="tests.envs.utils_envs:NoHuman",
)
gymnasium.register(
    id="test/NoHumanOldAPI-v0",
    entry_point="tests.envs.utils_envs:NoHumanOldAPI",
)

gymnasium.register(
    id="test/NoHumanNoRGB-v0",
    entry_point="tests.envs.utils_envs:NoHumanNoRGB",
)


def test_make():
    env = gymnasium.make("CartPole-v1", disable_env_checker=True)
    assert env.spec.id == "CartPole-v1"
    assert isinstance(env.unwrapped, cartpole.CartPoleEnv)
    env.close()


def test_make_deprecated():
    with warnings.catch_warnings(record=True):
        with pytest.raises(
            gymnasium.error.Error,
            match=re.escape(
                "Environment version v0 for `Humanoid` is deprecated. Please use `Humanoid-v4` instead."
            ),
        ):
            gymnasium.make("Humanoid-v0", disable_env_checker=True)


def test_make_max_episode_steps():
    # Default, uses the spec's
    env = gymnasium.make("CartPole-v1", disable_env_checker=True)
    assert has_wrapper(env, TimeLimit)
    assert (
            env.spec.max_episode_steps == gymnasium.envs.registry["CartPole-v1"].max_episode_steps
    )
    env.close()

    # Custom max episode steps
    env = gymnasium.make("CartPole-v1", max_episode_steps=100, disable_env_checker=True)
    assert has_wrapper(env, TimeLimit)
    assert env.spec.max_episode_steps == 100
    env.close()

    # Env spec has no max episode steps
    assert gymnasium.spec("test.ArgumentEnv-v0").max_episode_steps is None
    env = gymnasium.make(
        "test.ArgumentEnv-v0", arg1=None, arg2=None, arg3=None, disable_env_checker=True
    )
    assert has_wrapper(env, TimeLimit) is False
    env.close()


def test_gym_make_autoreset():
    """Tests that `gymnasium.make` autoreset wrapper is applied only when `gymnasium.make(..., autoreset=True)`."""
    env = gymnasium.make("CartPole-v1", disable_env_checker=True)
    assert has_wrapper(env, AutoResetWrapper) is False
    env.close()

    env = gymnasium.make("CartPole-v1", autoreset=False, disable_env_checker=True)
    assert has_wrapper(env, AutoResetWrapper) is False
    env.close()

    env = gymnasium.make("CartPole-v1", autoreset=True)
    assert has_wrapper(env, AutoResetWrapper)
    env.close()


def test_make_disable_env_checker():
    """Tests that `gymnasium.make` disable env checker is applied only when `gymnasium.make(..., disable_env_checker=False)`."""
    spec = deepcopy(gymnasium.spec("CartPole-v1"))

    # Test with spec disable env checker
    spec.disable_env_checker = False
    env = gymnasium.make(spec)
    assert has_wrapper(env, PassiveEnvChecker)
    env.close()

    # Test with overwritten spec using make disable env checker
    assert spec.disable_env_checker is False
    env = gymnasium.make(spec, disable_env_checker=True)
    assert has_wrapper(env, PassiveEnvChecker) is False
    env.close()

    # Test with spec enabled disable env checker
    spec.disable_env_checker = True
    env = gymnasium.make(spec)
    assert has_wrapper(env, PassiveEnvChecker) is False
    env.close()

    # Test with overwritten spec using make disable env checker
    assert spec.disable_env_checker is True
    env = gymnasium.make(spec, disable_env_checker=False)
    assert has_wrapper(env, PassiveEnvChecker)
    env.close()


def test_apply_api_compatibility():
    gymnasium.register(
        "testing-old-env",
        lambda: GenericTestEnv(step_fn=old_step_fn),
        apply_api_compatibility=True,
        max_episode_steps=3,
    )
    env = gymnasium.make("testing-old-env")

    env.reset()
    assert len(env.step(env.action_space.sample())) == 5
    env.step(env.action_space.sample())
    _, _, termination, truncation, _ = env.step(env.action_space.sample())
    assert termination is False and truncation is True

    gymnasium.spec("testing-old-env").apply_api_compatibility = False
    env = gymnasium.make("testing-old-env")
    # Cannot run reset and step as will not work

    env = gymnasium.make("testing-old-env", apply_api_compatibility=True)

    env.reset()
    assert len(env.step(env.action_space.sample())) == 5
    env.step(env.action_space.sample())
    _, _, termination, truncation, _ = env.step(env.action_space.sample())
    assert termination is False and truncation is True

    gymnasium.envs.registry.pop("testing-old-env")


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_passive_checker_wrapper_warnings(spec):
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = gymnasium.make(spec)  # disable_env_checker=False
        env.reset()
        env.step(env.action_space.sample())
        # todo, add check for render, bugged due to mujoco v2/3 and v4 envs

        env.close()

    for warning in caught_warnings:
        if warning.message.args[0] not in PASSIVE_CHECK_IGNORE_WARNING:
            raise gymnasium.error.Error(f"Unexpected warning: {warning.message}")


def test_make_order_enforcing():
    """Checks that gymnasium.make wrappers the environment with the OrderEnforcing wrapper."""
    assert all(spec.order_enforce is True for spec in all_testing_env_specs)

    env = gymnasium.make("CartPole-v1", disable_env_checker=True)
    assert has_wrapper(env, OrderEnforcing)
    # We can assume that there all other specs will also have the order enforcing
    env.close()

    gymnasium.register(
        id="test.OrderlessArgumentEnv-v0",
        entry_point="tests.envs.utils_envs:ArgumentEnv",
        order_enforce=False,
        kwargs={"arg1": None, "arg2": None, "arg3": None},
    )

    env = gymnasium.make("test.OrderlessArgumentEnv-v0", disable_env_checker=True)
    assert has_wrapper(env, OrderEnforcing) is False
    env.close()


def test_make_render_mode():
    env = gymnasium.make("CartPole-v1", disable_env_checker=True)
    assert env.render_mode is None
    env.close()

    # Make sure that render_mode is applied correctly
    env = gymnasium.make(
        "CartPole-v1", render_mode="rgb_array_list", disable_env_checker=True
    )
    assert env.render_mode == "rgb_array_list"
    env.reset()
    renders = env.render()
    assert isinstance(
        renders, list
    )  # Make sure that the `render` method does what is supposed to
    assert isinstance(renders[0], np.ndarray)
    env.close()

    env = gymnasium.make("CartPole-v1", render_mode=None, disable_env_checker=True)
    assert env.render_mode is None
    valid_render_modes = env.metadata["render_modes"]
    env.close()

    assert len(valid_render_modes) > 0
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = gymnasium.make(
            "CartPole-v1", render_mode=valid_render_modes[0], disable_env_checker=True
        )
        assert env.render_mode == valid_render_modes[0]
        env.close()

    for warning in caught_warnings:
        raise gymnasium.error.Error(f"Unexpected warning: {warning.message}")

    # Make sure that native rendering is used when possible
    env = gymnasium.make("CartPole-v1", render_mode="human", disable_env_checker=True)
    assert not has_wrapper(env, HumanRendering)  # Should use native human-rendering
    assert env.render_mode == "human"
    env.close()

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "You are trying to use 'human' rendering for an environment that doesn't natively support it. The HumanRendering wrapper is being applied to your environment."
        ),
    ):
        # Make sure that `HumanRendering` is applied here
        env = gymnasium.make(
            "test/NoHuman-v0", render_mode="human", disable_env_checker=True
        )  # This environment doesn't use native rendering
        assert has_wrapper(env, HumanRendering)
        assert env.render_mode == "human"
        env.close()

    with pytest.raises(
        TypeError, match=re.escape("got an unexpected keyword argument 'render_mode'")
    ):
        gymnasium.make(
            "test/NoHumanOldAPI-v0",
            render_mode="rgb_array_list",
            disable_env_checker=True,
        )

    # Make sure that an additional error is thrown a user tries to use the wrapper on an environment with old API
    with warnings.catch_warnings(record=True):
        with pytest.raises(
            gymnasium.error.Error,
            match=re.escape(
                "You passed render_mode='human' although test/NoHumanOldAPI-v0 doesn't implement human-rendering natively."
            ),
        ):
            gymnasium.make(
                "test/NoHumanOldAPI-v0", render_mode="human", disable_env_checker=True
            )

    # This test ensures that the additional exception "Gym tried to apply the HumanRendering wrapper but it looks like
    # your environment is using the old rendering API" is *not* triggered by a TypeError that originate from
    # a keyword that is not `render_mode`
    with pytest.raises(
        TypeError,
        match=re.escape("got an unexpected keyword argument 'render'"),
    ):
        gymnasium.make("CarRacing-v2", render="human")


def test_make_kwargs():
    env = gymnasium.make(
        "test.ArgumentEnv-v0",
        arg2="override_arg2",
        arg3="override_arg3",
        disable_env_checker=True,
    )
    assert env.spec.id == "test.ArgumentEnv-v0"
    assert isinstance(env.unwrapped, ArgumentEnv)
    assert env.arg1 == "arg1"
    assert env.arg2 == "override_arg2"
    assert env.arg3 == "override_arg3"
    env.close()


def test_import_module_during_make():
    # Test custom environment which is registered at make
    env = gymnasium.make(
        "tests.envs.utils:RegisterDuringMakeEnv-v0",
        disable_env_checker=True,
    )
    assert isinstance(env.unwrapped, RegisterDuringMakeEnv)
    env.close()
