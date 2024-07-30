import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import VectorEnv


@pytest.mark.parametrize("num_envs", (1, 3))
def test_record_episode_statistics(num_envs, env_id="CartPole-v1", num_steps=100):
    wrapper_vector_env: VectorEnv = gym.wrappers.vector.RecordEpisodeStatistics(
        gym.make_vec(id=env_id, num_envs=num_envs, vectorization_mode="sync"),
    )
    vector_wrapper_env = gym.make_vec(
        id=env_id,
        num_envs=num_envs,
        vectorization_mode="sync",
        wrappers=(gym.wrappers.RecordEpisodeStatistics,),
    )

    assert wrapper_vector_env.action_space == vector_wrapper_env.action_space
    assert wrapper_vector_env.observation_space == vector_wrapper_env.observation_space
    assert (
        wrapper_vector_env.single_action_space == vector_wrapper_env.single_action_space
    )
    assert (
        wrapper_vector_env.single_observation_space
        == vector_wrapper_env.single_observation_space
    )

    assert wrapper_vector_env.num_envs == vector_wrapper_env.num_envs

    wrapper_vector_obs, wrapper_vector_info = wrapper_vector_env.reset(seed=123)
    vector_wrapper_obs, vector_wrapper_info = vector_wrapper_env.reset(seed=123)

    assert data_equivalence(wrapper_vector_obs, vector_wrapper_obs)
    assert data_equivalence(wrapper_vector_info, vector_wrapper_info)

    for t in range(1, num_steps + 1):
        action = wrapper_vector_env.action_space.sample()
        (
            wrapper_vector_obs,
            wrapper_vector_reward,
            wrapper_vector_terminated,
            wrapper_vector_truncated,
            wrapper_vector_info,
        ) = wrapper_vector_env.step(action)
        (
            vector_wrapper_obs,
            vector_wrapper_reward,
            vector_wrapper_terminated,
            vector_wrapper_truncated,
            vector_wrapper_info,
        ) = vector_wrapper_env.step(action)

        assert data_equivalence(wrapper_vector_obs, vector_wrapper_obs)
        assert data_equivalence(wrapper_vector_reward, vector_wrapper_reward)
        assert data_equivalence(wrapper_vector_terminated, vector_wrapper_terminated)
        assert data_equivalence(wrapper_vector_truncated, vector_wrapper_truncated)

        if "episode" in wrapper_vector_info:
            wrapper_vector_time = wrapper_vector_info["episode"].pop("t")
            vector_wrapper_time = vector_wrapper_info["episode"].pop("t")
            assert wrapper_vector_time.shape == vector_wrapper_time.shape
            assert wrapper_vector_time.dtype == vector_wrapper_time.dtype

            vector_wrapper_info["episode"].pop("_l")
            vector_wrapper_info["episode"].pop("_r")
            vector_wrapper_info["episode"].pop("_t")

        assert data_equivalence(wrapper_vector_info, vector_wrapper_info)

    wrapper_vector_env.close()
    vector_wrapper_env.close()
