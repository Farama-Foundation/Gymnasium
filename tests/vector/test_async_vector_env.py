import re
from multiprocessing import TimeoutError

import numpy as np
import pytest

from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    NoAsyncCallError,
)
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from tests.vector.utils import (
    CustomSpace,
    make_custom_space_env,
    make_env,
    make_slow_env,
)


@pytest.mark.parametrize("shared_memory", [True, False])
def test_create_async_vector_env(shared_memory):
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    assert env.num_envs == 8
    env.close()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_reset_async_vector_env(shared_memory):
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    observations, infos = env.reset()

    env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert isinstance(infos, dict)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape

    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        observations, infos = env.reset()
    finally:
        env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape
    assert isinstance(infos, dict)
    assert all([isinstance(info, dict) for info in infos])


@pytest.mark.parametrize("shared_memory", [True, False])
@pytest.mark.parametrize("use_single_action_space", [True, False])
def test_step_async_vector_env(shared_memory, use_single_action_space):
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    observations = env.reset()

    assert isinstance(env.single_action_space, Discrete)
    assert isinstance(env.action_space, MultiDiscrete)

    if use_single_action_space:
        actions = [env.single_action_space.sample() for _ in range(8)]
    else:
        actions = env.action_space.sample()
    observations, rewards, terminateds, truncateds, _ = env.step(actions)

    env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape

    assert isinstance(rewards, np.ndarray)
    assert isinstance(rewards[0], (float, np.floating))
    assert rewards.ndim == 1
    assert rewards.size == 8

    assert isinstance(terminateds, np.ndarray)
    assert terminateds.dtype == np.bool_
    assert terminateds.ndim == 1
    assert terminateds.size == 8

    assert isinstance(truncateds, np.ndarray)
    assert truncateds.dtype == np.bool_
    assert truncateds.ndim == 1
    assert truncateds.size == 8


@pytest.mark.parametrize("shared_memory", [True, False])
def test_call_async_vector_env(shared_memory):
    env_fns = [
        make_env("CartPole-v1", i, render_mode="rgb_array_list") for i in range(4)
    ]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    _ = env.reset()
    images = env.call("render")
    gravity = env.call("gravity")

    env.close()

    assert isinstance(images, tuple)
    assert len(images) == 4
    for i in range(4):
        assert len(images[i]) == 1
        assert isinstance(images[i][0], np.ndarray)

    assert isinstance(gravity, tuple)
    assert len(gravity) == 4
    for i in range(4):
        assert isinstance(gravity[i], float)
        assert gravity[i] == 9.8


@pytest.mark.parametrize("shared_memory", [True, False])
def test_set_attr_async_vector_env(shared_memory):
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    env.set_attr("gravity", [9.81, 3.72, 8.87, 1.62])
    gravity = env.get_attr("gravity")
    assert gravity == (9.81, 3.72, 8.87, 1.62)

    env.close()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_copy_async_vector_env(shared_memory):
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]

    # TODO, these tests do nothing, understand the purpose of the tests and fix them
    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory, copy=True)
    observations, infos = env.reset()
    observations[0] = 0

    env.close()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_no_copy_async_vector_env(shared_memory):
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]

    # TODO, these tests do nothing, understand the purpose of the tests and fix them
    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory, copy=False)
    observations, infos = env.reset()
    observations[0] = 0

    env.close()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_reset_timeout_async_vector_env(shared_memory):
    env_fns = [make_slow_env(0.3, i) for i in range(4)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    with pytest.raises(TimeoutError):
        env.reset_async()
        env.reset_wait(timeout=0.1)

    env.close(terminate=True)


@pytest.mark.parametrize("shared_memory", [True, False])
def test_step_timeout_async_vector_env(shared_memory):
    env_fns = [make_slow_env(0.0, i) for i in range(4)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    with pytest.raises(TimeoutError):
        env.reset()
        env.step_async(np.array([0.1, 0.1, 0.3, 0.1]))
        observations, rewards, terminateds, truncateds, _ = env.step_wait(timeout=0.1)
    env.close(terminate=True)


@pytest.mark.parametrize("shared_memory", [True, False])
def test_reset_out_of_order_async_vector_env(shared_memory):
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    with pytest.raises(
        NoAsyncCallError,
        match=re.escape(
            "Calling `reset_wait` without any prior call to `reset_async`."
        ),
    ):
        env.reset_wait()

    env.close(terminate=True)

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    with pytest.raises(
        AlreadyPendingCallError,
        match=re.escape(
            "Calling `reset_async` while waiting for a pending call to `step` to complete"
        ),
    ):
        actions = env.action_space.sample()
        env.reset()
        env.step_async(actions)
        env.reset_async()

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Calling `close` while waiting for a pending call to `step` to complete."
        ),
    ):
        env.close(terminate=True)


@pytest.mark.parametrize("shared_memory", [True, False])
def test_step_out_of_order_async_vector_env(shared_memory):
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    with pytest.raises(
        NoAsyncCallError,
        match=re.escape("Calling `step_wait` without any prior call to `step_async`."),
    ):
        env.action_space.sample()
        env.reset()
        env.step_wait()

    env.close(terminate=True)

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    with pytest.raises(
        AlreadyPendingCallError,
        match=re.escape(
            "Calling `step_async` while waiting for a pending call to `reset` to complete"
        ),
    ):
        actions = env.action_space.sample()
        env.reset_async()
        env.step_async(actions)

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Calling `close` while waiting for a pending call to `reset` to complete."
        ),
    ):
        env.close(terminate=True)


@pytest.mark.parametrize("shared_memory", [True, False])
def test_already_closed_async_vector_env(shared_memory):
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]
    with pytest.raises(ClosedEnvironmentError):
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        env.close()
        env.reset()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_check_spaces_async_vector_env(shared_memory):
    # CartPole-v1 - observation_space: Box(4,), action_space: Discrete(2)
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]
    # FrozenLake-v1 - Discrete(16), action_space: Discrete(4)
    env_fns[1] = make_env("FrozenLake-v1", 1)
    with pytest.raises(RuntimeError):
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        env.close(terminate=True)


def test_custom_space_async_vector_env():
    env_fns = [make_custom_space_env(i) for i in range(4)]

    env = AsyncVectorEnv(env_fns, shared_memory=False)
    reset_observations, reset_infos = env.reset()

    assert isinstance(env.single_action_space, CustomSpace)
    assert isinstance(env.action_space, Tuple)

    actions = ("action-2", "action-3", "action-5", "action-7")
    step_observations, rewards, terminateds, truncateds, _ = env.step(actions)

    env.close()

    assert isinstance(env.single_observation_space, CustomSpace)
    assert isinstance(env.observation_space, Tuple)

    assert isinstance(reset_observations, tuple)
    assert reset_observations == ("reset", "reset", "reset", "reset")

    assert isinstance(step_observations, tuple)
    assert step_observations == (
        "step(action-2)",
        "step(action-3)",
        "step(action-5)",
        "step(action-7)",
    )


def test_custom_space_async_vector_env_shared_memory():
    env_fns = [make_custom_space_env(i) for i in range(4)]
    with pytest.raises(ValueError):
        env = AsyncVectorEnv(env_fns, shared_memory=True)
        env.close(terminate=True)
