"""Test suite for lambda observation wrappers: """

import numpy as np

import gymnasium as gym
from gymnasium.experimental.wrappers import (
    DtypeObservationV0,
    FilterObservationV0,
    FlattenObservationV0,
    GrayscaleObservationV0,
    LambdaObservationV0,
    RescaleObservationV0,
    ReshapeObservationV0,
    ResizeObservationV0,
)
from gymnasium.spaces import Box, Dict, Tuple
from tests.testing_env import GenericTestEnv


SEED = 42


def _record_random_obs_reset(self: gym.Env, seed=None, options=None):
    obs = self.observation_space.sample()
    return obs, {"obs": obs}


def _record_random_obs_step(self: gym.Env, action):
    obs = self.observation_space.sample()
    return obs, 0, False, False, {"obs": obs}


def _record_action_obs_reset(self: gym.Env, seed=None, options: dict = {}):
    return options["obs"], {"obs": options["obs"]}


def _record_action_obs_step(self: gym.Env, action):
    return action, 0, False, False, {"obs": action}


def _check_obs(
    env: gym.Env,
    wrapped_env: gym.Wrapper,
    transformed_obs,
    original_obs,
    strict: bool = True,
):
    assert (
        transformed_obs in wrapped_env.observation_space
    ), f"{transformed_obs}, {wrapped_env.observation_space}"
    assert (
        original_obs in env.observation_space
    ), f"{original_obs}, {env.observation_space}"

    if strict:
        assert (
            transformed_obs not in env.observation_space
        ), f"{transformed_obs}, {env.observation_space}"
        assert (
            original_obs not in wrapped_env.observation_space
        ), f"{original_obs}, {wrapped_env.observation_space}"


def test_lambda_observation_wrapper():
    """Tests lambda observation that the function is applied to both the reset and step observation."""
    env = GenericTestEnv(
        reset_func=_record_action_obs_reset, step_func=_record_action_obs_step
    )
    wrapped_env = LambdaObservationV0(env, lambda obs: obs + 2, Box(2, 3))

    obs, info = wrapped_env.reset(options={"obs": np.array([0], dtype=np.float32)})
    _check_obs(env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(np.array([1], dtype=np.float32))
    _check_obs(env, wrapped_env, obs, info["obs"])


def test_filter_observation_wrapper():
    """Tests ``FilterObservation`` that the right keys are filtered."""
    dict_env = GenericTestEnv(
        observation_space=Dict(arm_1=Box(0, 1), arm_2=Box(2, 3), arm_3=Box(-1, 1)),
        reset_func=_record_random_obs_reset,
        step_func=_record_random_obs_step,
    )

    wrapped_env = FilterObservationV0(dict_env, ("arm_1", "arm_3"))
    obs, info = wrapped_env.reset()
    assert list(obs.keys()) == ["arm_1", "arm_3"]
    assert list(info["obs"].keys()) == ["arm_1", "arm_2", "arm_3"]
    _check_obs(dict_env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(None)
    assert list(obs.keys()) == ["arm_1", "arm_3"]
    assert list(info["obs"].keys()) == ["arm_1", "arm_2", "arm_3"]
    _check_obs(dict_env, wrapped_env, obs, info["obs"])

    # Test tuple environments
    tuple_env = GenericTestEnv(
        observation_space=Tuple((Box(0, 1), Box(2, 3), Box(-1, 1))),
        reset_func=_record_random_obs_reset,
        step_func=_record_random_obs_step,
    )
    wrapped_env = FilterObservationV0(tuple_env, (2,))

    obs, info = wrapped_env.reset()
    assert len(obs) == 1 and len(info["obs"]) == 3
    _check_obs(tuple_env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(None)
    assert len(obs) == 1 and len(info["obs"]) == 3
    _check_obs(tuple_env, wrapped_env, obs, info["obs"])


def test_flatten_observation_wrapper():
    """Tests the ``FlattenObservation`` wrapper that the observation are flattened correctly."""
    env = GenericTestEnv(
        observation_space=Dict(arm=Box(0, 1), head=Box(2, 3)),
        reset_func=_record_random_obs_reset,
        step_func=_record_random_obs_step,
    )
    print(env.observation_space)
    wrapped_env = FlattenObservationV0(env)
    print(wrapped_env.observation_space)

    obs, info = wrapped_env.reset()
    _check_obs(env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(None)
    _check_obs(env, wrapped_env, obs, info["obs"])


def test_grayscale_observation_wrapper():
    """Tests the ``GrayscaleObservation`` that the observation is grayscale."""
    env = GenericTestEnv(
        observation_space=Box(0, 255, shape=(25, 25, 3), dtype=np.uint8),
        reset_func=_record_random_obs_reset,
        step_func=_record_random_obs_step,
    )
    wrapped_env = GrayscaleObservationV0(env)

    obs, info = wrapped_env.reset()
    _check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == (25, 25)

    obs, _, _, _, info = wrapped_env.step(None)
    _check_obs(env, wrapped_env, obs, info["obs"])

    # Keep_dim
    wrapped_env = GrayscaleObservationV0(env, keep_dim=True)

    obs, info = wrapped_env.reset()
    _check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == (25, 25, 1)

    obs, _, _, _, info = wrapped_env.step(None)
    _check_obs(env, wrapped_env, obs, info["obs"])


def test_resize_observation_wrapper():
    """Test the ``ResizeObservation`` that the observation has changed size"""
    env = GenericTestEnv(
        observation_space=Box(0, 255, shape=(60, 60, 3), dtype=np.uint8),
        reset_func=_record_random_obs_reset,
        step_func=_record_random_obs_step,
    )
    wrapped_env = ResizeObservationV0(env, (25, 25))

    obs, info = wrapped_env.reset()
    _check_obs(env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(None)
    _check_obs(env, wrapped_env, obs, info["obs"])


def test_reshape_observation_wrapper():
    """Test the ``ReshapeObservation`` wrapper."""
    env = GenericTestEnv(
        observation_space=Box(0, 1, shape=(2, 3, 2)),
        reset_func=_record_random_obs_reset,
        step_func=_record_random_obs_step,
    )
    wrapped_env = ReshapeObservationV0(env, (6, 2))

    obs, info = wrapped_env.reset()
    _check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == (6, 2)

    obs, _, _, _, info = wrapped_env.step(None)
    _check_obs(env, wrapped_env, obs, info["obs"])
    assert obs.shape == (6, 2)


def test_rescale_observation():
    """Test the ``RescaleObservation`` wrapper"""
    env = GenericTestEnv(
        observation_space=Box(
            np.array([0, 1], dtype=np.float32), np.array([1, 3], dtype=np.float32)
        ),
        reset_func=_record_action_obs_reset,
        step_func=_record_action_obs_step,
    )
    wrapped_env = RescaleObservationV0(
        env,
        min_obs=np.array([-5, 0], dtype=np.float32),
        max_obs=np.array([5, 1], dtype=np.float32),
    )
    assert wrapped_env.observation_space == Box(
        np.array([-5, 0], dtype=np.float32), np.array([5, 1], dtype=np.float32)
    )

    for sample_obs, expected_obs in (
        (
            np.array([0.5, 2.0], dtype=np.float32),
            np.array([0.0, 0.5], dtype=np.float32),
        ),
        (
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([-5.0, 0.0], dtype=np.float32),
        ),
        (
            np.array([1.0, 3.0], dtype=np.float32),
            np.array([5.0, 1.0], dtype=np.float32),
        ),
    ):
        assert sample_obs in env.observation_space
        assert expected_obs in wrapped_env.observation_space

        obs, info = wrapped_env.reset(options={"obs": sample_obs})
        assert np.all(obs == expected_obs)
        _check_obs(env, wrapped_env, obs, info["obs"], strict=False)

        obs, _, _, _, info = wrapped_env.step(sample_obs)
        assert np.all(obs == expected_obs)
        _check_obs(env, wrapped_env, obs, info["obs"], strict=False)


def test_dtype_observation():
    """Test ``DtypeObservation`` that the"""
    env = GenericTestEnv(
        reset_func=_record_random_obs_reset, step_func=_record_random_obs_step
    )
    wrapped_env = DtypeObservationV0(env, dtype=np.uint8)

    obs, info = wrapped_env.reset()
    assert obs.dtype != info["obs"].dtype
    assert obs.dtype == np.uint8

    obs, _, _, _, info = wrapped_env.step(None)
    assert obs.dtype != info["obs"].dtype
    assert obs.dtype == np.uint8
