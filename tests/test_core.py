"""Checks that the core Gymnasium API is implemented as expected."""

from __future__ import annotations

import re
from typing import Any, SupportsFloat

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import ActionWrapper, Env, ObservationWrapper, RewardWrapper, Wrapper
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from gymnasium.utils.seeding import np_random
from gymnasium.wrappers import OrderEnforcing
from tests.testing_env import GenericTestEnv


class ExampleEnv(Env):
    """Example testing environment."""

    def __init__(self):
        """Constructor for example environment."""
        self.observation_space = Box(0, 1)
        self.action_space = Box(0, 1)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the environment."""
        return 0, 0, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[ObsType, dict]:
        """Resets the environment."""
        super().reset(seed=seed, options=options)
        return 0, {}


@pytest.fixture
def example_env():
    return ExampleEnv()


def test_example_env(example_env):
    """Tests a gymnasium environment."""

    assert example_env.metadata == {"render_modes": []}
    assert example_env.render_mode is None
    assert example_env.spec is None
    assert example_env._np_random is None  # pyright: ignore [reportPrivateUsage]


class ExampleWrapper(Wrapper):
    """An example testing wrapper."""

    def __init__(self, env: Env[ObsType, ActType]):
        """Constructor that sets the reward."""
        super().__init__(env)

        self.new_reward = 3

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment ."""
        return super().reset(seed=seed, options=options)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the environment."""
        obs, reward, termination, truncation, info = self.env.step(action)
        return obs, self.new_reward, termination, truncation, info

    def access_hidden_np_random(self):
        """This should raise an error when called as wrappers should not access their own `_np_random` instances and should use the unwrapped environments."""
        return self._np_random


def test_example_wrapper(example_env):
    """Tests the gymnasium wrapper works as expected."""
    env = example_env
    wrapper_env = ExampleWrapper(env)

    assert env.metadata == wrapper_env.metadata
    wrapper_env.metadata = {"render_modes": ["rgb_array"]}
    assert env.metadata != wrapper_env.metadata

    assert env.render_mode == wrapper_env.render_mode

    assert env.spec == wrapper_env.spec

    env.observation_space = Box(0, 1)
    env.action_space = Box(0, 1)
    assert env.observation_space == wrapper_env.observation_space
    assert env.action_space == wrapper_env.action_space
    wrapper_env.observation_space = Box(1, 2)
    wrapper_env.action_space = Box(1, 2)
    assert env.observation_space != wrapper_env.observation_space
    assert env.action_space != wrapper_env.action_space

    wrapper_env.np_random, _ = seeding.np_random()
    assert (
        env._np_random  # pyright: ignore [reportPrivateUsage]
        is env.np_random
        is wrapper_env.np_random
    )
    assert 0 <= wrapper_env.np_random.uniform() <= 1
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`."
        ),
    ):
        _ = wrapper_env.access_hidden_np_random()


class ExampleRewardWrapper(RewardWrapper):
    """Example reward wrapper for testing."""

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Reward function."""
        return 1


class ExampleObservationWrapper(ObservationWrapper):
    """Example observation wrapper for testing."""

    def observation(self, observation: ObsType) -> ObsType:
        """Observation function."""
        return np.array([1])


class ExampleActionWrapper(ActionWrapper):
    """Example action wrapper for testing."""

    def action(self, action: ActType) -> ActType:
        """Action function."""
        return np.array([1])


def test_reward_observation_action_wrapper():
    """Tests the observation, action and reward wrapper examples."""
    env = GenericTestEnv()

    reward_env = ExampleRewardWrapper(env)
    reward_env.reset()
    _, reward, _, _, _ = reward_env.step(0)
    assert reward == 1

    observation_env = ExampleObservationWrapper(env)
    obs, _ = observation_env.reset()
    assert obs == np.array([1])
    obs, _, _, _, _ = observation_env.step(0)
    assert obs == np.array([1])

    env = GenericTestEnv(step_func=lambda self, action: (action, 0, False, False, {}))
    action_env = ExampleActionWrapper(env)
    obs, _, _, _, _ = action_env.step(0)
    assert obs == np.array([1])


def test_get_set_wrapper_attr():
    env = gym.make("CartPole-v1")
    assert env is not env.unwrapped

    # Test get_wrapper_attr
    with pytest.raises(AttributeError):
        env.gravity
    assert env.unwrapped.gravity is not None
    assert env.has_wrapper_attr("gravity")
    assert env.get_wrapper_attr("gravity") is not None

    with pytest.raises(AttributeError):
        env.unknown_attr
    assert env.has_wrapper_attr("unknown_attr") is False
    with pytest.raises(AttributeError):
        env.get_wrapper_attr("unknown_attr")

    # Test set_wrapper_attr
    env.set_wrapper_attr("gravity", 10.0)
    with pytest.raises(AttributeError):
        env.gravity  # checks the top level wrapper hasn't been updated
    assert env.unwrapped.gravity == 10.0
    assert env.get_wrapper_attr("gravity") == 10.0

    env.gravity = 5.0
    assert env.gravity == 5.0
    assert env.get_wrapper_attr("gravity") == 5.0
    assert env.env.get_wrapper_attr("gravity") == 10.0

    # Test with OrderEnforcing (intermediate wrapper)
    assert not isinstance(env, OrderEnforcing)

    # show that the base and top level objects don't contain the attribute
    with pytest.raises(AttributeError):
        env._disable_render_order_enforcing
    with pytest.raises(AttributeError):
        env.unwrapped._disable_render_order_enforcing
    assert env.has_wrapper_attr("_disable_render_order_enforcing")
    assert env.get_wrapper_attr("_disable_render_order_enforcing") is False

    env.set_wrapper_attr("_disable_render_order_enforcing", True)

    with pytest.raises(AttributeError):
        env._disable_render_order_enforcing
    with pytest.raises(AttributeError):
        env.unwrapped._disable_render_order_enforcing
    assert env.get_wrapper_attr("_disable_render_order_enforcing") is True

    # Test with top-most wrapper
    env.MY_ATTRIBUTE_1 = True
    assert env.get_wrapper_attr("MY_ATTRIBUTE_1") is True
    env.set_wrapper_attr("MY_ATTRIBUTE_1", False)
    assert env.get_wrapper_attr("MY_ATTRIBUTE_1") is False

    # Test with non-existing attribute
    env.set_wrapper_attr("MY_ATTRIBUTE_2", True)
    assert getattr(env, "MY_ATTRIBUTE_2") is True


class TestRandomSeeding:
    @staticmethod
    def test_nonempty_seed_retrieved_when_not_set(example_env):
        assert example_env.np_random_seed is not None
        assert isinstance(example_env.np_random_seed, int)

    @staticmethod
    def test_seed_set_at_reset_and_retrieved(example_env):
        seed = 42
        example_env.reset(seed=seed)
        assert example_env.np_random_seed == seed
        # resetting with seed=None means seed remains the same
        example_env.reset(seed=None)
        assert example_env.np_random_seed == seed

    @staticmethod
    def test_seed_cannot_be_set_directly(example_env):
        with pytest.raises(AttributeError):
            example_env.np_random_seed = 42

    @staticmethod
    def test_negative_seed_retrieved_when_seed_unknown(example_env):
        rng, _ = np_random()
        example_env.np_random = rng
        # seed is unknown
        assert example_env.np_random_seed == -1

    @staticmethod
    def test_seeding_works_in_wrapped_envs(example_env):
        seed = 42
        wrapper_env = ExampleWrapper(example_env)
        wrapper_env.reset(seed=seed)
        assert wrapper_env.np_random_seed == seed
        # resetting with seed=None means seed remains the same
        wrapper_env.reset(seed=None)
        assert wrapper_env.np_random_seed == seed
        # setting np_random directly makes seed unknown
        rng, _ = np_random()
        wrapper_env.np_random = rng
        assert wrapper_env.np_random_seed == -1
