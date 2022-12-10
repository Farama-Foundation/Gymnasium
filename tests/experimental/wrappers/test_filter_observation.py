"""Test suite for FilterObservationV0."""
from gymnasium.experimental.wrappers import FilterObservationV0
from gymnasium.spaces import Box, Dict, Tuple
from tests.experimental.wrappers.utils import (
    check_obs,
    record_random_obs_reset,
    record_random_obs_step,
)
from tests.testing_env import GenericTestEnv


def test_filter_observation_wrapper():
    """Tests ``FilterObservation`` that the right keys are filtered."""
    dict_env = GenericTestEnv(
        observation_space=Dict(arm_1=Box(0, 1), arm_2=Box(2, 3), arm_3=Box(-1, 1)),
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )

    wrapped_env = FilterObservationV0(dict_env, ("arm_1", "arm_3"))
    obs, info = wrapped_env.reset()
    assert list(obs.keys()) == ["arm_1", "arm_3"]
    assert list(info["obs"].keys()) == ["arm_1", "arm_2", "arm_3"]
    check_obs(dict_env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(None)
    assert list(obs.keys()) == ["arm_1", "arm_3"]
    assert list(info["obs"].keys()) == ["arm_1", "arm_2", "arm_3"]
    check_obs(dict_env, wrapped_env, obs, info["obs"])

    # Test tuple environments
    tuple_env = GenericTestEnv(
        observation_space=Tuple((Box(0, 1), Box(2, 3), Box(-1, 1))),
        reset_func=record_random_obs_reset,
        step_func=record_random_obs_step,
    )
    wrapped_env = FilterObservationV0(tuple_env, (2,))

    obs, info = wrapped_env.reset()
    assert len(obs) == 1 and len(info["obs"]) == 3
    check_obs(tuple_env, wrapped_env, obs, info["obs"])

    obs, _, _, _, info = wrapped_env.step(None)
    assert len(obs) == 1 and len(info["obs"]) == 3
    check_obs(tuple_env, wrapped_env, obs, info["obs"])
