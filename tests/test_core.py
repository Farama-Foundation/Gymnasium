from typing import Any, Dict, Optional, SupportsFloat, Tuple

import numpy as np

from gymnasium import Env, ObservationWrapper, RewardWrapper, Wrapper
from gymnasium.core import ActionWrapper, ActType, ObsType
from gymnasium.spaces import Box
from tests.generic_test_env import GenericTestEnv


class ExampleEnv(Env):
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
    pass


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
