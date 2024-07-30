"""Test the `SyncVectorEnv` implementation."""

import re
import warnings
from multiprocessing import TimeoutError

import numpy as np
import pytest

from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    NoAsyncCallError,
)
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple
from gymnasium.vector import AsyncVectorEnv
from tests.testing_env import GenericTestEnv
from tests.vector.testing_utils import (
    CustomSpace,
    make_custom_space_env,
    make_env,
    make_slow_env,
)


@pytest.mark.parametrize("shared_memory", [True, False])
def test_create_async_vector_env(shared_memory):
    """Test creating an async vector environment with or without shared memory."""
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    assert env.num_envs == 8
    env.close()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_reset_async_vector_env(shared_memory):
    """Test the reset of async vector environment with or without shared memory."""
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    observations, infos = env.reset()

    env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
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


def test_render_async_vector():
    envs = AsyncVectorEnv(
        [make_env("CartPole-v1", i, render_mode="rgb_array") for i in range(3)]
    )
    assert envs.render_mode == "rgb_array"

    envs.reset()
    rendered_frames = envs.render()
    assert isinstance(rendered_frames, tuple)
    assert len(rendered_frames) == envs.num_envs
    assert all(isinstance(frame, np.ndarray) for frame in rendered_frames)
    envs.close()

    envs = AsyncVectorEnv([make_env("CartPole-v1", i) for i in range(3)])
    assert envs.render_mode is None
    envs.close()


@pytest.mark.parametrize("shared_memory", [True, False])
@pytest.mark.parametrize("use_single_action_space", [True, False])
def test_step_async_vector_env(shared_memory, use_single_action_space):
    """Test the step async vector environment with and without shared memory."""
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    env.reset()

    assert isinstance(env.single_action_space, Discrete)
    assert isinstance(env.action_space, MultiDiscrete)

    if use_single_action_space:
        actions = [env.single_action_space.sample() for _ in range(8)]
    else:
        actions = env.action_space.sample()
    observations, rewards, terminations, truncations, _ = env.step(actions)

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

    assert isinstance(terminations, np.ndarray)
    assert terminations.dtype == np.bool_
    assert terminations.ndim == 1
    assert terminations.size == 8

    assert isinstance(truncations, np.ndarray)
    assert truncations.dtype == np.bool_
    assert truncations.ndim == 1
    assert truncations.size == 8


@pytest.mark.parametrize("shared_memory", [True, False])
def test_call_async_vector_env(shared_memory):
    """Test call with async vector environment."""
    env_fns = [
        make_env("CartPole-v1", i, render_mode="rgb_array_list") for i in range(4)
    ]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    env.reset()
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
    """Test `set_attr_` for async vector environment with or without shared memory."""
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    env.set_attr("gravity", [9.81, 3.72, 8.87, 1.62])
    gravity = env.get_attr("gravity")
    assert gravity == (9.81, 3.72, 8.87, 1.62)

    env.close()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_copy_async_vector_env(shared_memory):
    """Test observations are a copy of the true observation with and without shared memory."""
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]

    # TODO, these tests do nothing, understand the purpose of the tests and fix them
    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory, copy=True)
    observations, infos = env.reset()
    observations[0] = 0

    env.close()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_no_copy_async_vector_env(shared_memory):
    """Test observation are not a copy of the true observation with and without shared memory."""
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]

    # TODO, these tests do nothing, understand the purpose of the tests and fix them
    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory, copy=False)
    observations, infos = env.reset()
    observations[0] = 0

    env.close()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_reset_timeout_async_vector_env(shared_memory):
    """Test timeout error on reset with and without shared memory."""
    env_fns = [make_slow_env(0.3, i) for i in range(4)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    with pytest.raises(TimeoutError):
        env.reset_async()
        env.reset_wait(timeout=0.1)

    env.close(terminate=True)


@pytest.mark.parametrize("shared_memory", [True, False])
def test_step_timeout_async_vector_env(shared_memory):
    """Test timeout error on step with and without shared memory."""
    env_fns = [make_slow_env(0.0, i) for i in range(4)]

    env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    with pytest.raises(TimeoutError):
        env.reset()
        env.step_async(np.array([0.1, 0.1, 0.3, 0.1]))
        observations, rewards, terminations, truncations, _ = env.step_wait(timeout=0.1)
    env.close(terminate=True)


@pytest.mark.parametrize("shared_memory", [True, False])
def test_reset_out_of_order_async_vector_env(shared_memory):
    """Test reset being called out of order with and without shared memory."""
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
    """Test step out of order with and without shared memory."""
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
    """Test the error if a function is called if environment is already closed."""
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]
    with pytest.raises(ClosedEnvironmentError):
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        env.close()
        env.reset()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_check_spaces_async_vector_env(shared_memory):
    """Test check spaces for async vector environment with and without shared memory."""
    # CartPole-v1 - observation_space: Box(4,), action_space: Discrete(2)
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]
    # FrozenLake-v1 - Discrete(16), action_space: Discrete(4)
    env_fns[1] = make_env("FrozenLake-v1", 1)
    with pytest.raises(RuntimeError):
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        env.close(terminate=True)


def test_custom_space_async_vector_env():
    """Test custom spaces with async vector environment."""
    env_fns = [make_custom_space_env(i) for i in range(4)]

    env = AsyncVectorEnv(env_fns, shared_memory=False)
    reset_observations, reset_infos = env.reset()

    assert isinstance(env.single_action_space, CustomSpace)
    assert isinstance(env.action_space, Tuple)

    actions = ("action-2", "action-3", "action-5", "action-7")
    step_observations, rewards, terminations, truncations, _ = env.step(actions)

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
    """Test custom space with shared memory."""
    env_fns = [make_custom_space_env(i) for i in range(4)]
    with pytest.raises(ValueError):
        env = AsyncVectorEnv(env_fns, shared_memory=True)
        env.close(terminate=True)


def raise_error_reset(self, seed, options):
    super(GenericTestEnv, self).reset(seed=seed, options=options)
    if seed == 1:
        raise ValueError("Error in reset")
    return self.observation_space.sample(), {}


def raise_error_step(self, action):
    if action >= 1:
        raise ValueError(f"Error in step with {action}")

    return self.observation_space.sample(), 0, False, False, {}


def test_async_vector_subenv_error():
    envs = AsyncVectorEnv(
        [
            lambda: GenericTestEnv(
                reset_func=raise_error_reset, step_func=raise_error_step
            )
        ]
        * 2
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        envs.reset(seed=[0, 0])
    assert len(caught_warnings) == 0

    with warnings.catch_warnings(record=True) as caught_warnings:
        with pytest.raises(ValueError, match="Error in reset"):
            envs.reset(seed=[1, 0])

    envs.close()

    assert len(caught_warnings) == 3
    assert (
        "Received the following error from Worker-0 - Shutting it down"
        in caught_warnings[0].message.args[0]
    )
    assert (
        'in raise_error_reset\n    raise ValueError("Error in reset")\nValueError: Error in reset'
        in caught_warnings[1].message.args[0]
    )
    assert (
        caught_warnings[2].message.args[0]
        == "\x1b[31mERROR: Raising the last exception back to the main process.\x1b[0m"
    )

    envs = AsyncVectorEnv(
        [
            lambda: GenericTestEnv(
                reset_func=raise_error_reset, step_func=raise_error_step
            )
        ]
        * 3
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        with pytest.raises(ValueError, match="Error in step"):
            envs.step([0, 1, 2])

    envs.close()

    assert len(caught_warnings) == 5
    # due to variance in the step time, the order of warnings is random
    assert re.match(
        r"\x1b\[31mERROR: Received the following error from Worker-[12] - Shutting it down\x1b\[0m",
        caught_warnings[0].message.args[0],
    )
    assert re.match(
        r"\x1b\[31mERROR: Traceback \(most recent call last\):(?s:.)*in raise_error_step(?s:.)*ValueError: Error in step with [12]\n\x1b\[0m",
        caught_warnings[1].message.args[0],
    )
    assert re.match(
        r"\x1b\[31mERROR: Received the following error from Worker-[12] - Shutting it down\x1b\[0m",
        caught_warnings[2].message.args[0],
    )
    assert re.match(
        r"\x1b\[31mERROR: Traceback \(most recent call last\):(?s:.)*in raise_error_step(?s:.)*ValueError: Error in step with [12]\n\x1b\[0m",
        caught_warnings[3].message.args[0],
    )
    assert (
        caught_warnings[4].message.args[0]
        == "\x1b[31mERROR: Raising the last exception back to the main process.\x1b[0m"
    )
