"""Test suite for RecordEpisodeStatistics wrapper."""

import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers import Autoreset, RecordEpisodeStatistics
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
@pytest.mark.parametrize("deque_size", [2, 5])
def test_record_episode_statistics(env_id, deque_size):
    env = gym.make(env_id, disable_env_checker=True)
    env = RecordEpisodeStatistics(env, deque_size)

    for _ in range(5):
        env.reset()
        assert env.episode_returns is not None and env.episode_lengths is not None
        assert env.episode_returns == 0.0
        assert env.episode_lengths == 0
        assert env.spec is not None
        for _ in range(env.spec.max_episode_steps):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                assert "episode" in info
                assert all([item in info["episode"] for item in ["r", "l", "t"]])
                break
    assert len(env.return_queue) == deque_size
    assert len(env.length_queue) == deque_size


def _fixed_length_env(episode_length, reward, metadata=None):
    def reset_func(self, seed=None, options=None):
        self.timestep = 0
        return self.observation_space.sample(), {}

    def step_func(self, action):
        self.timestep += 1
        return (
            self.observation_space.sample(),
            reward,
            self.timestep >= episode_length,
            False,
            {},
        )

    return GenericTestEnv(reset_func=reset_func, step_func=step_func, metadata=metadata)


@pytest.mark.parametrize(
    "autoreset_mode", [AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP]
)
def test_record_episode_statistics_with_autoreset(
    autoreset_mode, env_id="CartPole-v1", num_steps=300
):
    """Tests that `RecordEpisodeStatistics(Autoreset(env))` is equivalent to `Autoreset(RecordEpisodeStatistics(env))`.

    For `Autoreset(RecordEpisodeStatistics(env))`, the statistics wrapper is reset by `Autoreset`
    so the statistics are correct without knowing the autoreset mode.
    """
    wrapper_autoreset_env = RecordEpisodeStatistics(
        Autoreset(gym.make(env_id), autoreset_mode=autoreset_mode)
    )
    autoreset_wrapper_env = Autoreset(
        RecordEpisodeStatistics(gym.make(env_id)), autoreset_mode=autoreset_mode
    )

    wrapper_autoreset_obs, wrapper_autoreset_info = wrapper_autoreset_env.reset(
        seed=123
    )
    autoreset_wrapper_obs, autoreset_wrapper_info = autoreset_wrapper_env.reset(
        seed=123
    )
    assert data_equivalence(wrapper_autoreset_obs, autoreset_wrapper_obs)
    assert data_equivalence(wrapper_autoreset_info, autoreset_wrapper_info)

    wrapper_autoreset_env.action_space.seed(123)
    for _ in range(num_steps):
        action = wrapper_autoreset_env.action_space.sample()
        wrapper_autoreset_step = wrapper_autoreset_env.step(action)
        autoreset_wrapper_step = autoreset_wrapper_env.step(action)
        wrapper_autoreset_info = wrapper_autoreset_step[-1]
        autoreset_wrapper_info = autoreset_wrapper_step[-1]

        # For same-step autoreset, the inner statistics wrapper's episode statistics are in `final_info`
        # as the environment is reset within the step, while the outer wrapper adds them to `info`
        if (
            autoreset_mode == AutoresetMode.SAME_STEP
            and "final_info" in autoreset_wrapper_info
        ):
            autoreset_wrapper_info["episode"] = autoreset_wrapper_info[
                "final_info"
            ].pop("episode")

        # The episode time depends on when each wrapper measures it
        if "episode" in wrapper_autoreset_info:
            wrapper_autoreset_info["episode"].pop("t")
            autoreset_wrapper_info["episode"].pop("t")

        assert data_equivalence(wrapper_autoreset_step, autoreset_wrapper_step)

    stats_wrapper = autoreset_wrapper_env.env
    assert isinstance(stats_wrapper, RecordEpisodeStatistics)
    assert stats_wrapper.episode_count > 1
    assert wrapper_autoreset_env.episode_count == stats_wrapper.episode_count
    assert wrapper_autoreset_env.return_queue == stats_wrapper.return_queue
    assert wrapper_autoreset_env.length_queue == stats_wrapper.length_queue

    wrapper_autoreset_env.close()
    autoreset_wrapper_env.close()


def test_record_episode_statistics_invalid_autoreset_mode():
    env = _fixed_length_env(
        3, 1.0, metadata={"render_modes": [], "autoreset_mode": "next_step"}
    )
    with pytest.raises(TypeError, match="AutoresetMode"):
        RecordEpisodeStatistics(env)
