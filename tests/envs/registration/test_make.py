"""Tests that `gym.make` works as expected."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import Env
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.error import NameNotFound
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.wrappers import (
    HumanRendering,
    OrderEnforcing,
    PassiveEnvChecker,
    TimeLimit,
)
from tests.envs.registration.utils_envs import ArgumentEnv
from tests.envs.utils import all_testing_env_specs
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import has_wrapper


# Tests
#  * basic example
#  * parameters (equivalent for str and EnvSpec)
#   1. max_episode_steps
#   2. autoreset
#   3. apply_api_compatibility
#   4. disable_env_checker
#  * rendering
#   1. render_mode
#   2. HumanRendering
#   3. RenderCollection
#  * make kwargs
#  * make import module
#  * make env spec additional wrappers
#  * env_id str errors


def test_no_arguments(env_id: str = "CartPole-v1"):
    """Test `gym.make` using str and EnvSpec with no arguments."""
    env_from_id = gym.make(env_id)
    assert env_from_id.spec is not None
    assert env_from_id.spec.id == env_id
    assert isinstance(env_from_id.unwrapped, CartPoleEnv)

    env_spec = gym.spec(env_id)
    env_from_spec = gym.make(env_spec)
    assert env_from_spec.spec is not None
    assert env_from_spec.spec.id == env_id
    assert isinstance(env_from_spec.unwrapped, CartPoleEnv)

    assert env_from_id.spec == env_from_spec.spec


def test_max_episode_steps(register_parameter_envs):
    """Test the `max_episode_steps` parameter in `gym.make`."""
    for make_id in ["CartPole-v1", gym.spec("CartPole-v1")]:
        env_spec = gym.spec(make_id) if isinstance(make_id, str) else make_id

        # Use the spec's value
        env = gym.make(make_id)
        assert has_wrapper(env, TimeLimit)
        assert env.spec is not None
        assert env.spec.max_episode_steps == env_spec.max_episode_steps

        # Set a custom max episode steps value
        assert env_spec.max_episode_steps != 100
        env = gym.make(make_id, max_episode_steps=100)
        assert has_wrapper(env, TimeLimit)
        assert env.spec is not None
        assert env.spec.max_episode_steps == 100, make_id

    for make_id in ["NoMaxEpisodeStepsEnv-v0", gym.spec("NoMaxEpisodeStepsEnv-v0")]:
        env_spec = gym.spec(make_id) if isinstance(make_id, str) else make_id

        # env spec has no max episode steps
        assert env_spec.max_episode_steps is None
        env = gym.make(make_id)
        assert env.spec is not None
        assert env.spec.max_episode_steps is None
        assert has_wrapper(env, TimeLimit) is False

        # set a custom max episode steps values
        env = gym.make(make_id, max_episode_steps=100)
        assert env.spec is not None
        assert env.spec.max_episode_steps == 100
        assert has_wrapper(env, TimeLimit)

    # Override max_episode_step to prevent applying the wrapper
    for env_id in [
        "CartPole-v1",
        gym.spec("CartPole-v1"),
        "NoMaxEpisodeStepsEnv-v0",
        gym.spec("NoMaxEpisodeStepsEnv-v0"),
    ]:
        env = gym.make(env_id, max_episode_steps=-1)
        assert env.spec is not None
        assert env.spec.max_episode_steps is None
        assert has_wrapper(env, TimeLimit) is False


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
def test_disable_env_checker(
    registration_disabled: bool, make_disabled: bool | None, if_disabled: bool
):
    """Tests that `gym.make` disable env checker is applied only when `gym.make(..., disable_env_checker=False)`.

    The ordering is 1. if the `make(..., disable_env_checker=...)` is bool, then the `registration(..., disable_env_checker=...)`
    """
    gym.register(
        "DisableEnvCheckerEnv-v0",
        lambda: GenericTestEnv(),
        disable_env_checker=registration_disabled,
    )

    # Test when the registered EnvSpec.disable_env_checker = False
    env = gym.make("DisableEnvCheckerEnv-v0", disable_env_checker=make_disabled)
    assert has_wrapper(env, PassiveEnvChecker) is not if_disabled

    env_spec = gym.spec("DisableEnvCheckerEnv-v0")
    env = gym.make(env_spec, disable_env_checker=make_disabled)
    assert has_wrapper(env, PassiveEnvChecker) is not if_disabled

    del gym.registry["DisableEnvCheckerEnv-v0"]


def test_order_enforcing(register_parameter_envs):
    """Checks that gym.make wrappers the environment with the OrderEnforcing wrapper."""
    assert all(spec.order_enforce is False for spec in all_testing_env_specs)

    for make_id in ["CartPole-v1", gym.spec("CartPole-v1")]:
        env = gym.make(make_id)
        assert has_wrapper(env, OrderEnforcing)

    for make_id in ["OrderlessEnv-v0", gym.spec("OrderlessEnv-v0")]:
        env = gym.make(make_id)
        assert has_wrapper(env, OrderEnforcing) is False

    # There is no `make(..., order_enforcing=...)` so we don't test that


def test_make_with_render_mode():
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


def test_make_human_rendering(register_rendering_testing_envs):
    # Make sure that native rendering is used when possible
    env = gym.make("CartPole-v1", render_mode="human")
    assert (
        has_wrapper(env, HumanRendering) is False
    )  # Should use native human-rendering
    assert env.render_mode == "human"
    env.close()

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "You are trying to use 'human' rendering for an environment that doesn't natively support it. The HumanRendering wrapper is being applied to your environment."
        ),
    ):
        # Make sure that `HumanRendering` is applied here as the environment doesn't use native rendering
        env = gym.make("NoHumanRendering-v0", render_mode="human")
        assert has_wrapper(env, HumanRendering)
        assert env.render_mode == "human"
        env.close()

    with pytest.raises(
        TypeError, match=re.escape("got an unexpected keyword argument 'render_mode'")
    ):
        gym.make(
            "NoHumanRenderingOldAPI-v0",
            render_mode="rgb_array_list",
        )

    # Make sure that an additional error is thrown a user tries to use the wrapper on an environment with old API
    with warnings.catch_warnings(record=True):
        with pytest.raises(
            gym.error.Error,
            match=re.escape(
                "You passed render_mode='human' although NoHumanRenderingOldAPI-v0 doesn't implement human-rendering natively."
            ),
        ):
            gym.make("NoHumanRenderingOldAPI-v0", render_mode="human")

    # This test ensures that the additional exception "Gym tried to apply the HumanRendering wrapper but it looks like
    # your environment is using the old rendering API" is *not* triggered by a TypeError that originate from
    # a keyword that is not `render_mode`
    with pytest.raises(
        TypeError,
        match=re.escape("got an unexpected keyword argument 'render'"),
    ):
        gym.make("CarRacing-v3", render="human")

    # This test checks that a user can create an environment without the metadata including the render mode
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "\x1b[33mWARN: The environment is being initialised with render_mode='rgb_array' that is not in the possible render_modes ([]).\x1b[0m"
        ),
    ):
        gym.make("NoRenderModesMetadata-v0", render_mode="rgb_array")


def test_make_kwargs(register_kwargs_env):
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
    assert env.unwrapped.arg1 == "arg1"
    assert env.unwrapped.arg2 == "override_arg2"
    assert env.unwrapped.arg3 == "override_arg3"
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


class NoRecordArgsWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

    def observation(self, observation: ObsType) -> WrapperObsType:
        return self.observation_space.sample()


def test_make_with_env_spec():
    # make
    id_env = gym.make("CartPole-v1")
    spec_env = gym.make(gym.spec("CartPole-v1"))
    assert id_env.spec == spec_env.spec

    # make with applied wrappers
    env_2 = gym.wrappers.NormalizeReward(
        gym.wrappers.TimeAwareObservation(
            gym.wrappers.FlattenObservation(
                gym.make("CartPole-v1", render_mode="rgb_array")
            )
        ),
        gamma=0.8,
    )
    env_2_recreated = gym.make(env_2.spec)
    assert env_2.spec == env_2_recreated.spec
    env_2.close()
    env_2_recreated.close()

    # make with callable entry point
    gym.register("CartPole-v2", lambda: CartPoleEnv())
    env_3 = gym.make("CartPole-v2")
    assert isinstance(env_3.unwrapped, CartPoleEnv)
    env_3.close()

    # make with wrapper in env-creator
    gym.register(
        "CartPole-v3",
        lambda: gym.wrappers.NormalizeReward(CartPoleEnv()),
        disable_env_checker=True,
        order_enforce=False,
    )
    env_4 = gym.make(gym.spec("CartPole-v3"))
    assert isinstance(env_4, gym.wrappers.NormalizeReward)
    assert isinstance(env_4.env, CartPoleEnv)
    env_4.close()

    gym.register(
        "CartPole-v4",
        lambda: CartPoleEnv(),
        disable_env_checker=True,
        order_enforce=False,
        additional_wrappers=(gym.wrappers.NormalizeReward.wrapper_spec(),),
    )
    env_5 = gym.make(gym.spec("CartPole-v4"))
    assert isinstance(env_5, gym.wrappers.NormalizeReward)
    assert isinstance(env_5.env, CartPoleEnv)
    env_5.close()

    # make with no ezpickle wrapper
    env_6 = NoRecordArgsWrapper(gym.make("CartPole-v1"))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "NoRecordArgsWrapper wrapper does not inherit from `gymnasium.utils.RecordConstructorArgs`, therefore, the wrapper cannot be recreated."
        ),
    ):
        gym.make(env_6.spec)

    # make with no ezpickle wrapper but in the entry point
    gym.register(
        "CartPole-v5",
        entry_point=lambda: NoRecordArgsWrapper(CartPoleEnv()),
        disable_env_checker=True,
        order_enforce=False,
    )
    env_7 = gym.make(gym.spec("CartPole-v5"))
    assert isinstance(env_7, NoRecordArgsWrapper)
    assert isinstance(env_7.unwrapped, CartPoleEnv)

    gym.register(
        "CartPole-v6",
        entry_point=lambda: CartPoleEnv(),
        disable_env_checker=True,
        order_enforce=False,
        additional_wrappers=(NoRecordArgsWrapper.wrapper_spec(),),
    )

    del gym.registry["CartPole-v2"]
    del gym.registry["CartPole-v3"]
    del gym.registry["CartPole-v4"]
    del gym.registry["CartPole-v5"]
    del gym.registry["CartPole-v6"]


def test_make_with_env_spec_levels():
    """Test that we can recreate the environment at each 'level'."""
    env = gym.wrappers.NormalizeReward(
        gym.wrappers.TimeAwareObservation(
            gym.wrappers.FlattenObservation(
                gym.make("CartPole-v1", render_mode="rgb_array")
            )
        ),
        gamma=0.8,
    )

    while env is not env.unwrapped:
        recreated_env = gym.make(env.spec)
        assert env.spec == recreated_env.spec

        env = env.env


def test_wrapped_env_entry_point():
    def _create_env():
        _env = gym.make("CartPole-v1", render_mode="rgb_array")
        _env = gym.wrappers.FlattenObservation(_env)
        return _env

    gym.register("TestingEnv-v0", entry_point=_create_env)

    env = gym.make("TestingEnv-v0")
    env = gym.wrappers.TimeAwareObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.8)

    recreated_env = gym.make(env.spec)

    obs, info = env.reset(seed=42)
    recreated_obs, recreated_info = recreated_env.reset(seed=42)
    assert data_equivalence(obs, recreated_obs)
    assert data_equivalence(info, recreated_info)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    (
        recreated_obs,
        recreated_reward,
        recreated_terminated,
        recreated_truncated,
        recreated_info,
    ) = recreated_env.step(action)
    assert data_equivalence(obs, recreated_obs)
    assert data_equivalence(reward, recreated_reward)
    assert data_equivalence(terminated, recreated_terminated)
    assert data_equivalence(truncated, recreated_truncated)
    assert data_equivalence(info, recreated_info)

    del gym.registry["TestingEnv-v0"]


def test_make_errors():
    """Test make with a deprecated environment (i.e., doesn't exist)."""
    with warnings.catch_warnings(record=True):
        with pytest.raises(
            gym.error.Error,
            match=re.escape(
                "Environment version v0 for `Humanoid` is deprecated. Please use `Humanoid-v5` instead."
            ),
        ):
            gym.make("Humanoid-v0")

    with pytest.raises(
        NameNotFound, match=re.escape("Environment `NonExistenceEnv` doesn't exist.")
    ):
        gym.make("NonExistenceEnv-v0")


@pytest.fixture(scope="function")
def register_parameter_envs():
    gym.register(
        "NoMaxEpisodeStepsEnv-v0", lambda: GenericTestEnv(), max_episode_steps=None
    )

    gym.register("OrderlessEnv-v0", lambda: GenericTestEnv(), order_enforce=False)

    yield

    del gym.registry["NoMaxEpisodeStepsEnv-v0"]
    del gym.registry["OrderlessEnv-v0"]


@pytest.fixture(scope="function")
def register_kwargs_env():
    gym.register(
        id="test.ArgumentEnv-v0",
        entry_point="tests.envs.registration.utils_envs:ArgumentEnv",
        kwargs={
            "arg1": "arg1",
            "arg2": "arg2",
        },
    )


@pytest.fixture(scope="function")
def register_rendering_testing_envs():
    gym.register(
        id="NoHumanRendering-v0",
        entry_point="tests.envs.registration.utils_envs:NoHuman",
    )
    gym.register(
        id="NoHumanRenderingOldAPI-v0",
        entry_point="tests.envs.registration.utils_envs:NoHumanOldAPI",
    )

    gym.register(
        id="NoHumanRenderingNoRGB-v0",
        entry_point="tests.envs.registration.utils_envs:NoHumanNoRGB",
    )

    gym.register(
        id="NoRenderModesMetadata-v0",
        entry_point="tests.envs.registration.utils_envs:NoRenderModesMetadata",
    )

    yield

    del gym.envs.registration.registry["NoHumanRendering-v0"]
    del gym.envs.registration.registry["NoHumanRenderingOldAPI-v0"]
    del gym.envs.registration.registry["NoHumanRenderingNoRGB-v0"]
    del gym.envs.registration.registry["NoRenderModesMetadata-v0"]
