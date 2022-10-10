"""Checks that the core Gymnasium API is implemented as expected."""
import re
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import numpy as np
import pytest

from gymnasium import Env, ObservationWrapper, RewardWrapper, Wrapper
from gymnasium.core import (
    ActionWrapper,
    ActType,
    ObsType,
    WrapperActType,
    WrapperObsType,
)
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from tests.testing_env import GenericTestEnv


class ExampleEnv(Env):
    def __init__(self):
        self.observation_space = Box(0, 1)
        self.action_space = Box(0, 1)

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        return 0, 0, False, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        return 0, {}


def test_gymnasium_env():
    env = ExampleEnv()

    assert env.metadata == {"render_modes": []}
    assert env.render_mode is None
    assert env.reward_range == (-float("inf"), float("inf"))
    assert env.spec is None
    assert env._np_random is None


class ExampleWrapper(Wrapper):
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        self.new_reward = 3

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def step(
        self, action: WrapperActType
    ) -> Tuple[WrapperObsType, float, bool, bool, Dict[str, Any]]:
        obs, reward, termination, truncation, info = self.env.step(action)
        return obs, self.new_reward, termination, truncation, info

    def access_hidden_np_random(self):
        """This should raise an error when called as wrappers should not access their own `_np_random` instances and should use the unwrapped environments."""
        return self._np_random


def test_gymnasium_wrapper():
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
        env._np_random is env.np_random is wrapper_env.np_random
    )  # ignore: reportPrivateUsage
    assert 0 <= wrapper_env.np_random.uniform() <= 1
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
        ),
    ):
        print(wrapper_env.access_hidden_np_random())


class ExampleRewardWrapper(RewardWrapper):
    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        return 1


class ExampleObservationWrapper(ObservationWrapper):
    def observation(self, observation: ObsType) -> ObsType:
        return np.array([1])


class ExampleActionWrapper(ActionWrapper):
    def action(self, action: ActType) -> ActType:
        return np.array([1])


def test_wrapper_types():
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

    env = GenericTestEnv(step_fn=lambda self, action: (action, 0, False, False, {}))
    action_env = ExampleActionWrapper(env)
    obs, _, _, _, _ = action_env.step(0)
    assert obs == np.array([1])
