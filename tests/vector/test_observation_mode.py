import re
from functools import partial

import numpy as np
import pytest

from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.vector.utils import batch_differing_spaces
from tests.testing_env import GenericTestEnv


def create_env(obs_space):
    return lambda: GenericTestEnv(observation_space=obs_space)


# Test cases for both SyncVectorEnv and AsyncVectorEnv
@pytest.mark.parametrize(
    "vector_env_fn",
    [SyncVectorEnv, AsyncVectorEnv, partial(AsyncVectorEnv, shared_memory=False)],
    ids=[
        "SyncVectorEnv",
        "AsyncVectorEnv(shared_memory=True)",
        "AsyncVectorEnv(shared_memory=False)",
    ],
)
class TestVectorEnvObservationModes:

    def test_invalid_observation_mode(self, vector_env_fn):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Invalid `observation_mode`, expected: 'same' or 'different' or tuple of single and batch observation space, actual got invalid"
            ),
        ):
            vector_env_fn(
                [create_env(Box(low=0, high=1, shape=(5,))) for _ in range(3)],
                observation_mode="invalid",
            )

    def test_obs_mode_same_different_spaces(self, vector_env_fn):
        spaces = [Box(low=0, high=i, shape=(2,)) for i in range(1, 4)]
        with pytest.raises(
            (AssertionError, RuntimeError),
            match="the sub-environments observation spaces are not equivalent. .*If this is intentional, use `observation_mode='different'` instead.",
        ):
            vector_env_fn(
                [create_env(space) for space in spaces], observation_mode="same"
            )

    @pytest.mark.parametrize(
        "observation_mode",
        [
            "different",
            (
                Box(
                    low=0,
                    high=np.repeat(np.arange(1, 4), 5).reshape((3, 5)),
                    shape=(3, 5),
                ),
                Box(low=0, high=1, shape=(5,)),
            ),
        ],
    )
    def test_obs_mode_different_different_spaces(self, vector_env_fn, observation_mode):
        spaces = [Box(low=0, high=i, shape=(5,)) for i in range(1, 4)]
        envs = vector_env_fn(
            [create_env(space) for space in spaces], observation_mode=observation_mode
        )
        assert envs.observation_space == batch_differing_spaces(spaces)
        assert envs.single_observation_space == spaces[0]

        envs.reset()
        envs.step(envs.action_space.sample())
        envs.close()

    @pytest.mark.parametrize(
        "observation_mode",
        [
            "different",
            (Box(low=0, high=4, shape=(3, 5)), Box(low=0, high=4, shape=(5,))),
        ],
    )
    def test_obs_mode_different_different_shapes(self, vector_env_fn, observation_mode):
        spaces = [Box(low=0, high=1, shape=(i + 1,)) for i in range(3)]
        with pytest.raises(
            (AssertionError, RuntimeError),
            # match=re.escape(
            #     "Expected all Box.low shape to be equal, actually [(1,), (2,), (3,)]"
            # ),
        ):
            vector_env_fn(
                [create_env(space) for space in spaces],
                observation_mode=observation_mode,
            )

    @pytest.mark.parametrize(
        "observation_mode",
        [
            "same",
            "different",
            (Box(low=0, high=4, shape=(3, 5)), Box(low=0, high=4, shape=(5,))),
        ],
    )
    def test_mixed_observation_spaces(self, vector_env_fn, observation_mode):
        spaces = [
            Box(low=0, high=1, shape=(3,)),
            Discrete(5),
            Dict({"a": Discrete(2), "b": Box(low=0, high=1, shape=(2,))}),
        ]

        with pytest.raises(
            (AssertionError, RuntimeError),
            # match=re.escape(
            #     "Expects all spaces to be the same shape, actual types: [<class 'gymnasium.spaces.box.Box'>, <class 'gymnasium.spaces.discrete.Discrete'>, <class 'gymnasium.spaces.dict.Dict'>]"
            # ),
        ):
            vector_env_fn(
                [create_env(space) for space in spaces],
                observation_mode=observation_mode,
            )
