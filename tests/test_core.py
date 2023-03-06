"""Checks that the core Gymnasium API is implemented as expected."""
import re
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import numpy as np
import pytest

from gymnasium import Env, ObservationWrapper, RewardWrapper, Wrapper, spaces
from gymnasium.core import (
    ActionWrapper,
    ActType,
    ObsType,
    WrapperActType,
    WrapperObsType,
)
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from gymnasium.wrappers import OrderEnforcing, TimeLimit
from tests.testing_env import GenericTestEnv


# ==== Old testing code


class ArgumentEnv(Env):
    """Testing environment that records the number of times the environment is created."""

    observation_space = spaces.Box(low=0, high=1, shape=(1,))
    action_space = spaces.Box(low=0, high=1, shape=(1,))
    calls = 0

    def __init__(self, arg: Any):
        """Constructor."""
        self.calls += 1
        self.arg = arg


class UnittestEnv(Env):
    """Example testing environment."""

    observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    action_space = spaces.Discrete(3)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the environment."""
        super().reset(seed=seed)
        return self.observation_space.sample(), {"info": "dummy"}

    def step(self, action):
        """Steps through the environment."""
        observation = self.observation_space.sample()  # Dummy observation
        return observation, 0.0, False, {}


class UnknownSpacesEnv(Env):
    """This environment defines its observation & action spaces only after the first call to reset.

    Although this pattern is sometimes necessary when implementing a new environment (e.g. if it depends
    on external resources), it is not encouraged.
    """

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the environment."""
        super().reset(seed=seed)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(3)
        return self.observation_space.sample(), {}  # Dummy observation with info

    def step(self, action):
        """Steps through the environment."""
        observation = self.observation_space.sample()  # Dummy observation
        return observation, 0.0, False, {}


class OldStyleEnv(Env):
    """This environment doesn't accept any arguments in reset, ideally we want to support this too (for now)."""

    def reset(self):
        """Resets the environment."""
        super().reset()
        return 0

    def step(self, action):
        """Steps through the environment."""
        return 0, 0, False, {}


class NewPropertyWrapper(Wrapper):
    """Wrapper that tests setting a property."""

    def __init__(
        self,
        env,
        observation_space=None,
        action_space=None,
        reward_range=None,
        metadata=None,
    ):
        """New property wrapper.

        Args:
            env: The environment to wrap
            observation_space: The observation space
            action_space: The action space
            reward_range: The reward range
            metadata: The environment metadata
        """
        super().__init__(env)
        if observation_space is not None:
            # Only set the observation space if not None to test property forwarding
            self.observation_space = observation_space
        if action_space is not None:
            self.action_space = action_space
        if reward_range is not None:
            self.reward_range = reward_range
        if metadata is not None:
            self.metadata = metadata


def test_env_instantiation():
    """Tests the environment instantiation using ArgumentEnv."""
    # This looks like a pretty trivial, but given our usage of
    # __new__, it's worth having.
    env = ArgumentEnv("arg")
    assert env.arg == "arg"
    assert env.calls == 1


properties = [
    {
        "observation_space": spaces.Box(
            low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32
        )
    },
    {"action_space": spaces.Discrete(2)},
    {"reward_range": (-1.0, 1.0)},
    {"metadata": {"render_modes": ["human", "rgb_array_list"]}},
    {
        "observation_space": spaces.Box(
            low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32
        ),
        "action_space": spaces.Discrete(2),
    },
]


@pytest.mark.parametrize("class_", [UnittestEnv, UnknownSpacesEnv])
@pytest.mark.parametrize("props", properties)
def test_wrapper_property_forwarding(class_, props):
    """Tests wrapper property forwarding."""
    env = class_()
    env = NewPropertyWrapper(env, **props)

    # If UnknownSpacesEnv, then call reset to define the spaces
    if isinstance(env.unwrapped, UnknownSpacesEnv):
        _ = env.reset()

    # Test the properties set by the wrapper
    for key, value in props.items():
        assert getattr(env, key) == value

    # Otherwise, test if the properties are forwarded
    all_properties = {"observation_space", "action_space", "reward_range", "metadata"}
    for key in all_properties - props.keys():
        assert getattr(env, key) == getattr(env.unwrapped, key)


def test_compatibility_with_old_style_env():
    """Test compatibility with old style environment."""
    env = OldStyleEnv()
    env = OrderEnforcing(env)
    env = TimeLimit(env, 100)
    obs = env.reset()
    assert obs == 0


# ==== New testing code


class ExampleEnv(Env):
    """Example testing environment."""

    def __init__(self):
        """Constructor for example environment."""
        self.observation_space = Box(0, 1)
        self.action_space = Box(0, 1)

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Steps through the environment."""
        return 0, 0, False, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """Resets the environment."""
        return 0, {}


def test_gymnasium_env():
    """Tests a gymnasium environment."""
    env = ExampleEnv()

    assert env.metadata == {"render_modes": []}
    assert env.render_mode is None
    assert env.reward_range == (-float("inf"), float("inf"))
    assert env.spec is None
    assert env._np_random is None  # pyright: ignore [reportPrivateUsage]


class ExampleWrapper(Wrapper):
    """An example testing wrapper."""

    def __init__(self, env: Env[ObsType, ActType]):
        """Constructor that sets the reward."""
        super().__init__(env)

        self.new_reward = 3

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        """Resets the environment ."""
        return super().reset(seed=seed, options=options)

    def step(
        self, action: WrapperActType
    ) -> Tuple[WrapperObsType, float, bool, bool, Dict[str, Any]]:
        """Steps through the environment."""
        obs, reward, termination, truncation, info = self.env.step(action)
        return obs, self.new_reward, termination, truncation, info

    def access_hidden_np_random(self):
        """This should raise an error when called as wrappers should not access their own `_np_random` instances and should use the unwrapped environments."""
        return self._np_random


def test_gymnasium_wrapper():
    """Tests the gymnasium wrapper works as expected."""
    env = ExampleEnv()
    wrapper_env = ExampleWrapper(env)

    assert env.metadata == wrapper_env.metadata
    wrapper_env.metadata = {"render_modes": ["rgb_array"]}
    assert env.metadata != wrapper_env.metadata

    assert env.render_mode == wrapper_env.render_mode

    assert env.reward_range == wrapper_env.reward_range
    wrapper_env.reward_range = (-1.0, 1.0)
    assert env.reward_range != wrapper_env.reward_range

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
            "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
        ),
    ):
        print(wrapper_env.access_hidden_np_random())


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


def test_wrapper_types():
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
