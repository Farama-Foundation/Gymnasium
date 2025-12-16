"""Testing of the `gym.make_vec` function."""

import multiprocessing
import re
import warnings

import pytest

import gymnasium as gym
from gymnasium import VectorizeMode, error, wrappers
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.classic_control.cartpole import CartPoleVectorEnv
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from gymnasium.wrappers import TimeLimit, TransformObservation
from tests.wrappers.utils import has_wrapper


def test_make_vec_env_id():
    """Ensure that the `gym.make_vec` creates the right environment."""
    env = gym.make_vec("CartPole-v1")
    assert isinstance(env, CartPoleVectorEnv)
    assert env.num_envs == 1
    env.close()


@pytest.mark.parametrize("num_envs", [1, 3, 10])
@pytest.mark.parametrize("vectorization_mode", ["vector_entry_point", "async", "sync"])
def test_make_vec_num_envs(num_envs, vectorization_mode):
    """Test that the `gym.make_vec` num_envs parameter works."""
    env = gym.make_vec(
        "CartPole-v1", num_envs=num_envs, vectorization_mode=vectorization_mode
    )
    assert env.num_envs == num_envs
    env.close()


def test_make_vec_vectorization_mode():
    """Tests the `gym.make_vec` vectorization mode works."""
    # Test the default value for spec with and without `vector_entry_point`
    env_spec = gym.spec("CartPole-v1")
    assert env_spec is not None and env_spec.vector_entry_point is not None
    env = gym.make_vec("CartPole-v1")
    assert isinstance(env, CartPoleVectorEnv)
    env.close()

    env_spec = gym.spec("Pendulum-v1")
    assert env_spec is not None and env_spec.vector_entry_point is None
    env = gym.make_vec("Pendulum-v1")
    assert isinstance(env, SyncVectorEnv)
    env.close()

    # Test `vector_entry_point` for env specs with and without it
    env = gym.make_vec("CartPole-v1", vectorization_mode="vector_entry_point")
    assert isinstance(env, CartPoleVectorEnv)
    env.close()

    env = gym.make_vec(
        "CartPole-v1", vectorization_mode=VectorizeMode.VECTOR_ENTRY_POINT
    )
    assert isinstance(env, CartPoleVectorEnv)
    env.close()

    with pytest.raises(
        gym.error.Error,
        match=re.escape(
            "Cannot create vectorized environment for Pendulum-v1 because it doesn't have a vector entry point defined."
        ),
    ):
        gym.make_vec("Pendulum-v1", vectorization_mode="vector_entry_point")

    # Test `async` and `sync`
    env = gym.make_vec("CartPole-v1", vectorization_mode="async")
    assert isinstance(env, AsyncVectorEnv)
    env.close()

    env = gym.make_vec("CartPole-v1", vectorization_mode=VectorizeMode.ASYNC)
    assert isinstance(env, AsyncVectorEnv)
    env.close()

    env = gym.make_vec("CartPole-v1", vectorization_mode="sync")
    assert isinstance(env, SyncVectorEnv)
    env.close()

    env = gym.make_vec("CartPole-v1", vectorization_mode=VectorizeMode.SYNC)
    assert isinstance(env, SyncVectorEnv)
    env.close()

    # Test environment with only a vector entry point and no entry point
    gym.register("VecOnlyEnv-v0", vector_entry_point=CartPoleVectorEnv)
    env_spec = gym.spec("VecOnlyEnv-v0")
    assert env_spec.entry_point is None and env_spec.vector_entry_point is not None

    with pytest.raises(
        gym.error.Error,
        match=re.escape(
            "Cannot create vectorized environment for VecOnlyEnv-v0 because it doesn't have an entry point defined."
        ),
    ):
        gym.make_vec("VecOnlyEnv-v0", vectorization_mode="async")
    del gym.registry["VecOnlyEnv-v0"]

    # Test with invalid vectorization mode
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid vectorization mode: 'invalid', valid modes: ['async', 'sync', 'vector_entry_point']"
        ),
    ):
        gym.make_vec("CartPole-v1", vectorization_mode="invalid")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid vectorization mode: 123, valid modes: ['async', 'sync', 'vector_entry_point']"
        ),
    ):
        gym.make_vec("CartPole-v1", vectorization_mode=123)


def test_make_vec_render_mode():
    envs = gym.make_vec(
        "CartPole-v1", vectorization_mode=VectorizeMode.VECTOR_ENTRY_POINT
    )
    assert envs.render_mode is None
    envs.close()

    envs = gym.make_vec(
        "CartPole-v1",
        render_mode="rgb_array",
        vectorization_mode=VectorizeMode.VECTOR_ENTRY_POINT,
    )
    assert envs.render_mode == "rgb_array"
    envs.close()


def test_make_vec_wrappers():
    """Tests that the `gym.make_vec` wrappers parameter works."""
    env = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")
    assert isinstance(env, SyncVectorEnv)
    assert len(env.envs) == 2

    sub_env = env.envs[0]
    assert isinstance(sub_env, gym.Env)
    assert sub_env.spec is not None
    if sub_env.spec.max_episode_steps is not None:
        assert has_wrapper(sub_env, TimeLimit)

    assert all(
        has_wrapper(sub_env, TransformObservation) is False for sub_env in env.envs
    )
    env.close()

    env = gym.make_vec(
        "CartPole-v1",
        num_envs=2,
        vectorization_mode="sync",
        wrappers=[
            lambda _env: TransformObservation(
                _env, lambda obs: obs * 2, sub_env.observation_space
            )
        ],
    )
    # As asynchronous environment are inaccessible, synchronous vector must be used
    assert isinstance(env, SyncVectorEnv)
    assert all(has_wrapper(sub_env, TransformObservation) for sub_env in env.envs)

    env.close()


@pytest.mark.parametrize(
    "env_id, kwargs",
    (
        ("CartPole-v1", {}),
        ("CartPole-v1", {"num_envs": 3}),
        ("CartPole-v1", {"vectorization_mode": "sync"}),
        ("CartPole-v1", {"vectorization_mode": "vector_entry_point"}),
        (
            "CartPole-v1",
            {"vectorization_mode": "sync", "vector_kwargs": {"copy": False}},
        ),
        (
            "CartPole-v1",
            {
                "vectorization_mode": "sync",
                "wrappers": (gym.wrappers.TimeAwareObservation,),
            },
        ),
        ("CartPole-v1", {"render_mode": "rgb_array"}),
        ("CartPole-v1", {"vectorization_mode": "sync", "max_episode_steps": 5}),
        ("CartPole-v1", {"sutton_barto_reward": True}),
        ("CartPole-v1", {"vectorization_mode": "sync", "sutton_barto_reward": True}),
        (gym.spec("CartPole-v1"), {}),
        (gym.spec("CartPole-v1"), {"num_envs": 3}),
        (gym.spec("CartPole-v1"), {"vectorization_mode": "sync"}),
        (gym.spec("CartPole-v1"), {"vectorization_mode": "vector_entry_point"}),
        (
            gym.spec("CartPole-v1"),
            {"vectorization_mode": "sync", "vector_kwargs": {"copy": False}},
        ),
        (
            gym.spec("CartPole-v1"),
            {
                "vectorization_mode": "sync",
                "wrappers": (gym.wrappers.TimeAwareObservation,),
            },
        ),
        (gym.spec("CartPole-v1"), {"render_mode": "rgb_array"}),
        (
            gym.spec("CartPole-v1"),
            {"vectorization_mode": "sync", "max_episode_steps": 5},
        ),
        (gym.spec("CartPole-v1"), {"sutton_barto_reward": True}),
        (
            gym.spec("CartPole-v1"),
            {"vectorization_mode": "sync", "sutton_barto_reward": True},
        ),
    ),
)
def test_make_vec_with_spec(env_id: str, kwargs: dict):
    envs = gym.make_vec(env_id, **kwargs)
    assert envs.spec is not None
    recreated_envs = gym.make_vec(envs.spec)

    # Assert equivalence
    assert envs.spec == recreated_envs.spec
    assert envs.num_envs == recreated_envs.num_envs

    assert type(envs) is type(recreated_envs)

    assert envs.observation_space == recreated_envs.observation_space
    assert envs.single_observation_space == recreated_envs.single_observation_space
    assert envs.action_space == recreated_envs.action_space
    assert envs.single_action_space == recreated_envs.single_action_space

    assert envs.render_mode == recreated_envs.render_mode

    envs.close()
    recreated_envs.close()


@pytest.mark.parametrize("ctx", [None] + multiprocessing.get_all_start_methods())
def test_async_with_dynamically_registered_env(ctx):
    gym.register("TestEnv-v0", CartPoleEnv)

    gym.make_vec(
        "TestEnv-v0", vectorization_mode="async", vector_kwargs=dict(context=ctx)
    )

    del gym.registry["TestEnv-v0"]


def test_make_vec_with_spec_additional_wrappers():
    gym.register(
        "TestEnv-v0",
        entry_point=CartPoleEnv,
        additional_wrappers=(
            wrappers.ClipReward.wrapper_spec(min_reward=-0.5, max_reward=0.5),
        ),
    )

    env = gym.make("TestEnv-v0")
    assert isinstance(env, wrappers.ClipReward)
    env.close()

    envs = gym.make_vec("TestEnv-v0")
    assert isinstance(envs.envs[0], wrappers.ClipReward)
    envs.close()

    gym.register(
        "TestEnv-v1",
        vector_entry_point=CartPoleVectorEnv,
        additional_wrappers=(
            wrappers.ClipReward.wrapper_spec(min_reward=-0.5, max_reward=0.5),
        ),
    )
    with pytest.raises(
        error.Error,
        match=re.escape(
            "Cannot use `vector_entry_point` vectorization mode with the additional_wrappers parameter in spec being not empty"
        ),
    ):
        gym.make_vec("TestEnv-v1")

    del gym.registry["TestEnv-v0"]
    del gym.registry["TestEnv-v1"]


class MissingMetadataVecEnv(VectorEnv):
    metadata = {"render_fps": 1}

    def __init__(self, num_envs: int):
        self.num_envs = num_envs


class IncorrectMetadataVecEnv(VectorEnv):
    metadata = {"autoreset_mode": "next_step"}

    def __init__(self, num_envs: int):
        self.num_envs = num_envs


def test_missing_autoreset_mode_metadata():
    gym.register("MissingMetadataVecEnv-v0", vector_entry_point=MissingMetadataVecEnv)
    gym.register(
        "IncorrectMetadataVecEnv-v0", vector_entry_point=IncorrectMetadataVecEnv
    )

    with warnings.catch_warnings():
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "The VectorEnv (MissingMetadataVecEnv(MissingMetadataVecEnv-v0, num_envs=1)) is missing AutoresetMode metadata, metadata={'render_fps': 1}"
            ),
        ):
            gym.make_vec("MissingMetadataVecEnv-v0")

        with pytest.warns(
            UserWarning,
            match=re.escape(
                "The VectorEnv (IncorrectMetadataVecEnv(IncorrectMetadataVecEnv-v0, num_envs=1)) metadata['autoreset_mode'] is not an instance of AutoresetMode, <class 'str'>."
            ),
        ):
            gym.make_vec("IncorrectMetadataVecEnv-v0")

    gym.registry.pop("MissingMetadataVecEnv-v0")
    gym.registry.pop("IncorrectMetadataVecEnv-v0")
