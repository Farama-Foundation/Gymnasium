from functools import partial

import numpy as np
import pytest

import gymnasium as gym
from gymnasium import VectorizeMode
from gymnasium.spaces import Discrete
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.vector.vector_env import AutoresetMode
from tests.testing_env import GenericTestEnv


def count_reset(
    self: GenericTestEnv, seed: int | None = None, options: dict | None = None
):
    super(GenericTestEnv, self).reset(seed=seed)

    self.count = seed if seed is not None else 0
    return self.count, {}


def count_step(self: GenericTestEnv, action):
    self.count += 1

    return self.count, action, self.count == self.max_count, False, {}


@pytest.mark.parametrize(
    "vectoriser",
    [
        SyncVectorEnv,
        AsyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=False),
    ],
    ids=["Sync", "Async(shared_memory=True)", "Async(shared_memory=False)"],
)
def test_autoreset_next_step(vectoriser):
    envs = vectoriser(
        [
            lambda: GenericTestEnv(
                action_space=Discrete(5),
                observation_space=Discrete(5),
                reset_func=count_reset,
                step_func=count_step,
            )
            for _ in range(3)
        ],
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )
    print(f"{envs.metadata=}")
    assert envs.metadata["autoreset_mode"] == AutoresetMode.NEXT_STEP
    envs.set_attr("max_count", [2, 3, 3])

    obs, info = envs.reset()
    assert np.all(obs == [0, 0, 0])
    assert info == {}

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [1, 1, 1])
    assert np.all(rewards == [1, 2, 3])
    assert np.all(terminations == [False, False, False])
    assert np.all(truncations == [False, False, False])
    assert info == {}

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [2, 2, 2])
    assert np.all(rewards == [1, 2, 3])
    assert np.all(terminations == [True, False, False])
    assert np.all(truncations == [False, False, False])
    assert info == {}

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [0, 3, 3])
    assert np.all(rewards == [0, 2, 3])
    assert np.all(terminations == [False, True, True])
    assert np.all(truncations == [False, False, False])
    assert info == {}

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [1, 0, 0])
    assert np.all(rewards == [1, 0, 0])
    assert np.all(terminations == [False, False, False])
    assert np.all(truncations == [False, False, False])
    assert info == {}

    envs.close()


@pytest.mark.parametrize(
    "vectoriser",
    [
        SyncVectorEnv,
        AsyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=False),
    ],
    ids=["Sync", "Async(shared_memory=True)", "Async(shared_memory=False)"],
)
def test_autoreset_within_step(vectoriser):
    envs = vectoriser(
        [
            lambda: GenericTestEnv(
                action_space=Discrete(5),
                observation_space=Discrete(5),
                reset_func=count_reset,
                step_func=count_step,
            )
            for _ in range(3)
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    assert envs.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    envs.set_attr("max_count", [2, 3, 3])

    obs, info = envs.reset()
    assert np.all(obs == [0, 0, 0])
    assert info == {}

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [1, 1, 1])
    assert np.all(rewards == [1, 2, 3])
    assert np.all(terminations == [False, False, False])
    assert np.all(truncations == [False, False, False])
    assert info == {}

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [0, 2, 2])
    assert np.all(rewards == [1, 2, 3])
    assert np.all(terminations == [True, False, False])
    assert np.all(truncations == [False, False, False])
    assert data_equivalence(
        info,
        {
            "final_obs": np.array([2, 0, 0]),
            "final_info": {},
            "_final_obs": np.array([True, False, False]),
            "_final_info": np.array([True, False, False]),
        },
    )

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [1, 0, 0])
    assert np.all(rewards == [1, 2, 3])
    assert np.all(terminations == [False, True, True])
    assert np.all(truncations == [False, False, False])
    assert data_equivalence(
        info,
        {
            "final_obs": np.array([0, 3, 3]),
            "final_info": {},
            "_final_obs": np.array([False, True, True]),
            "_final_info": np.array([False, True, True]),
        },
    )

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [0, 1, 1])
    assert np.all(rewards == [1, 2, 3])
    assert np.all(terminations == [True, False, False])
    assert np.all(truncations == [False, False, False])
    assert data_equivalence(
        info,
        {
            "final_obs": np.array([2, 0, 0]),
            "final_info": {},
            "_final_obs": np.array([True, False, False]),
            "_final_info": np.array([True, False, False]),
        },
    )

    envs.close()


@pytest.mark.parametrize(
    "vectoriser",
    [
        SyncVectorEnv,
        AsyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=False),
    ],
    ids=["Sync", "Async(shared_memory=True)", "Async(shared_memory=False)"],
)
def test_autoreset_disabled(vectoriser):
    envs = vectoriser(
        [
            lambda: GenericTestEnv(
                action_space=Discrete(5),
                observation_space=Discrete(5),
                reset_func=count_reset,
                step_func=count_step,
            )
            for _ in range(3)
        ],
        autoreset_mode=AutoresetMode.DISABLED,
    )
    assert envs.metadata["autoreset_mode"] == AutoresetMode.DISABLED
    envs.set_attr("max_count", [2, 3, 3])

    obs, info = envs.reset()
    assert np.all(obs == [0, 0, 0])
    assert info == {}

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [1, 1, 1])
    assert np.all(rewards == [1, 2, 3])
    assert np.all(terminations == [False, False, False])
    assert np.all(truncations == [False, False, False])
    assert info == {}

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [2, 2, 2])
    assert np.all(rewards == [1, 2, 3])
    assert np.all(terminations == [True, False, False])
    assert np.all(truncations == [False, False, False])
    assert info == {}

    obs, info = envs.reset(options={"mask": terminations})
    assert np.all(obs == [0, 2, 2])
    assert info == {}

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [1, 3, 3])
    assert np.all(rewards == [1, 2, 3])
    assert np.all(terminations == [False, True, True])
    assert np.all(truncations == [False, False, False])
    assert info == {}

    obs, info = envs.reset(options={"mask": terminations})
    assert np.all(obs == [1, 0, 0])
    assert info == {}

    obs, rewards, terminations, truncations, info = envs.step([1, 2, 3])
    assert np.all(obs == [2, 1, 1])
    assert np.all(rewards == [1, 2, 3])
    assert np.all(terminations == [True, False, False])
    assert np.all(truncations == [False, False, False])
    assert info == {}

    envs.close()


@pytest.mark.parametrize(
    "vectoriser",
    [
        SyncVectorEnv,
        AsyncVectorEnv,
        partial(AsyncVectorEnv, shared_memory=False),
    ],
    ids=["Sync", "Async(shared_memory=True)", "Async(shared_memory=False)"],
)
@pytest.mark.parametrize(
    "autoreset_mode",
    [AutoresetMode.NEXT_STEP, AutoresetMode.DISABLED, AutoresetMode.SAME_STEP],
)
def test_autoreset_metadata(vectoriser, autoreset_mode):
    envs = vectoriser(
        [lambda: GenericTestEnv(), lambda: GenericTestEnv()],
        autoreset_mode=autoreset_mode,
    )
    assert envs.metadata["autoreset_mode"] == autoreset_mode
    envs.close()

    envs = vectoriser(
        [lambda: GenericTestEnv(), lambda: GenericTestEnv()],
        autoreset_mode=autoreset_mode.value,
    )
    assert envs.metadata["autoreset_mode"] == autoreset_mode
    envs.close()


@pytest.mark.parametrize(
    "vectorization_mode", [VectorizeMode.SYNC, VectorizeMode.ASYNC]
)
@pytest.mark.parametrize(
    "autoreset_mode",
    [AutoresetMode.NEXT_STEP, AutoresetMode.DISABLED, AutoresetMode.SAME_STEP],
)
def test_make_vec_autoreset(vectorization_mode, autoreset_mode):
    envs = gym.make_vec(
        "CartPole-v1",
        vectorization_mode=vectorization_mode,
        vector_kwargs={"autoreset_mode": autoreset_mode},
    )
    envs.metadata["autoreset_mode"] = autoreset_mode
    envs.close()

    envs = gym.make_vec(
        "CartPole-v1",
        vectorization_mode=vectorization_mode,
        vector_kwargs={"autoreset_mode": autoreset_mode.value},
    )
    envs.metadata["autoreset_mode"] = autoreset_mode
    envs.close()
