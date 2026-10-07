import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AutoresetMode, SyncVectorEnv, VectorEnv
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
@pytest.mark.parametrize("num_envs", (1, 3))
def test_record_episode_statistics(
    autoreset_mode, num_envs, request, env_id="CartPole-v1", num_steps=100
):
    if autoreset_mode == AutoresetMode.DISABLED and num_envs > 1:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="`SyncVectorEnv.reset` pops `reset_mask` from the shared `options` dict, so the wrapper resets every sub-environment's statistics on a partial reset",
            )
        )

    wrapper_vector_env: VectorEnv = gym.wrappers.vector.RecordEpisodeStatistics(
        gym.make_vec(
            id=env_id,
            num_envs=num_envs,
            vectorization_mode="sync",
            vector_kwargs={"autoreset_mode": autoreset_mode},
        ),
    )
    vector_wrapper_env = gym.make_vec(
        id=env_id,
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": autoreset_mode},
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

    wrapper_vector_env.action_space.seed(123)
    num_episode_ends = 0
    for _ in range(1, num_steps + 1):
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

        # For same-step autoreset, the sub-environment's episode statistics are in `final_info`
        # as the sub-environment is reset within the step, while the vector wrapper adds them to `info`
        if (
            autoreset_mode == AutoresetMode.SAME_STEP
            and "final_info" in vector_wrapper_info
            and "episode" in vector_wrapper_info["final_info"]
        ):
            vector_wrapper_final_info = vector_wrapper_info["final_info"]
            vector_wrapper_info["episode"] = vector_wrapper_final_info.pop("episode")
            vector_wrapper_info["_episode"] = vector_wrapper_final_info.pop("_episode")

        if "episode" in wrapper_vector_info:
            wrapper_vector_time = wrapper_vector_info["episode"].pop("t")
            vector_wrapper_time = vector_wrapper_info["episode"].pop("t")
            assert wrapper_vector_time.shape == vector_wrapper_time.shape
            assert wrapper_vector_time.dtype == vector_wrapper_time.dtype

            vector_wrapper_info["episode"].pop("_l")
            vector_wrapper_info["episode"].pop("_r")
            vector_wrapper_info["episode"].pop("_t")

        assert data_equivalence(wrapper_vector_info, vector_wrapper_info)

        dones = np.logical_or(wrapper_vector_terminated, wrapper_vector_truncated)
        num_episode_ends += int(np.sum(dones))
        if autoreset_mode == AutoresetMode.DISABLED and np.any(dones):
            wrapper_vector_obs, wrapper_vector_info = wrapper_vector_env.reset(
                options={"reset_mask": dones}
            )
            vector_wrapper_obs, vector_wrapper_info = vector_wrapper_env.reset(
                options={"reset_mask": dones}
            )
            assert data_equivalence(wrapper_vector_obs, vector_wrapper_obs)
            assert data_equivalence(wrapper_vector_info, vector_wrapper_info)

    assert num_episode_ends >= num_envs
    assert wrapper_vector_env.episode_count == num_episode_ends

    wrapper_vector_env.close()
    vector_wrapper_env.close()


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
def test_episode_statistics_with_autoreset_mode(
    autoreset_mode, num_envs=2, episode_length=3, num_steps=25
):
    """Check that the recorded statistics are correct for all autoreset modes using fixed-length episodes."""

    def reset_func(self, seed=None, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}

    def step_func(self, action):
        self.timestep += 1
        return (
            self.observation_space.sample(),
            1.0,
            self.timestep >= episode_length,
            False,
            {},
        )

    envs = gym.wrappers.vector.RecordEpisodeStatistics(
        SyncVectorEnv(
            [
                lambda: GenericTestEnv(reset_func=reset_func, step_func=step_func)
                for _ in range(num_envs)
            ],
            autoreset_mode=autoreset_mode,
        )
    )

    envs.reset(seed=123)
    for _ in range(num_steps):
        actions = envs.action_space.sample()
        _, _, terminations, truncations, infos = envs.step(actions)

        dones = np.logical_or(terminations, truncations)
        if np.any(dones):
            assert "episode" in infos
            assert np.all(infos["episode"]["r"][dones] == episode_length)
            assert np.all(infos["episode"]["l"][dones] == episode_length)

            if autoreset_mode == AutoresetMode.DISABLED:
                envs.reset(options={"reset_mask": np.copy(dones)})

    assert envs.episode_count > num_envs
    assert np.all(np.array(envs.return_queue) == episode_length)
    assert np.all(np.array(envs.length_queue) == episode_length)
    envs.close()
