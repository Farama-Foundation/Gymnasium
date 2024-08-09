import pytest

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium.vector.utils.batched_spaces import batch_differing_spaces


class CustomEnv(gym.Env):
    def __init__(self, observation_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = Discrete(2)  # Dummy action space

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}


def create_env(obs_space):
    return lambda: CustomEnv(obs_space)


# Test cases for both SyncVectorEnv and AsyncVectorEnv
@pytest.mark.parametrize("VectorEnv", [SyncVectorEnv, AsyncVectorEnv])
class TestVectorEnvObservationModes:

    def test_invalid_observation_mode(self, VectorEnv):
        with pytest.raises(
            ValueError, match="Need to pass in mode for batching observations"
        ):
            VectorEnv(
                [create_env(Box(low=0, high=1, shape=(5,))) for _ in range(3)],
                observation_mode="invalid",
            )

    def test_mixed_observation_spaces(self, VectorEnv):
        spaces = [
            Box(low=0, high=1, shape=(3,)),
            Discrete(5),
            Dict({"a": Discrete(2), "b": Box(low=0, high=1, shape=(2,))}),
        ]
        with pytest.raises(
            AssertionError,
            match="Low & High values for observation spaces can be different but shapes need to be the same",
        ):
            VectorEnv(
                [create_env(space) for space in spaces], observation_mode="different"
            )

    def test_default_observation_mode(self, VectorEnv):
        single_space = Box(low=0, high=1, shape=(5,))
        env = VectorEnv(
            [create_env(single_space) for _ in range(3)]
        )  # No observation_mode specified
        assert env.observation_space == batch_space(single_space, 3)

    def test_different_observation_mode_same_shape(self, VectorEnv):
        spaces = [Box(low=0, high=i, shape=(5,)) for i in range(1, 4)]
        env = VectorEnv(
            [create_env(space) for space in spaces], observation_mode="different"
        )
        assert env.observation_space == batch_differing_spaces(spaces)

    def test_different_observation_mode_different_shapes(self, VectorEnv):
        spaces = [Box(low=0, high=1, shape=(i + 1,)) for i in range(3)]
        with pytest.raises(
            AssertionError,
            match="Low & High values for observation spaces can be different but shapes need to be the same",
        ):
            VectorEnv(
                [create_env(space) for space in spaces], observation_mode="different"
            )
