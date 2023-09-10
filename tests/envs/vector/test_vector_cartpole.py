import numpy as np

from gymnasium.envs.classic_control.cartpole import CartPoleEnv, CartPoleVectorEnv
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.wrappers import TimeLimit


def test_num_env_1_equivalence():
    env = CartPoleEnv()
    envs = CartPoleVectorEnv(num_envs=1)

    envs.action_space.seed(123)
    env_obs, env_info = env.reset(seed=123)
    envs_obs, envs_info = envs.reset(seed=123)
    assert data_equivalence(env_obs, envs_obs[0])
    assert data_equivalence(env_info, envs_info)

    episode_over = False
    while not episode_over:
        actions = envs.action_space.sample()

        env_obs, env_reward, env_terminated, env_truncated, env_info = env.step(
            actions[0]
        )
        envs_obs, envs_reward, envs_terminated, envs_truncated, envs_info = envs.step(
            actions
        )

        assert np.allclose(env_obs, envs_obs[0])
        assert env_reward == envs_reward[0]
        assert env_terminated == envs_terminated[0]
        assert env_truncated == envs_truncated[0]
        assert data_equivalence(env_info, envs_info)

        episode_over = env_terminated or env_truncated

    actions = envs.action_space.sample()
    env_obs, env_info = env.reset()
    envs_obs, envs_reward, envs_terminated, envs_truncated, envs_info = envs.step(
        actions
    )
    assert np.allclose(env_obs, envs_obs[0])
    assert np.all(envs_reward == np.array([0.0]))
    assert np.all(envs_terminated == np.array([False]))
    assert np.all(envs_truncated == np.array([False]))
    assert data_equivalence(env_info, envs_info)


def test_timelimit(time_limit=5):
    env = TimeLimit(CartPoleEnv(), time_limit)
    envs = CartPoleVectorEnv(num_envs=1, max_episode_steps=time_limit)

    envs.action_space.seed(123)
    env.reset(seed=123)
    envs.reset(seed=123)

    for _ in range(time_limit - 1):
        actions = envs.action_space.sample()
        _, _, env_terminated, env_truncated, _ = env.step(actions[0])
        _, _, envs_terminated, envs_truncated, _ = envs.step(actions)

        assert env_terminated is False and env_truncated is False
        assert (not np.any(envs_terminated)) and (not np.any(envs_truncated))

    actions = envs.action_space.sample()
    _, _, env_terminated, env_truncated, _ = env.step(actions[0])
    _, _, envs_terminated, envs_truncated, _ = envs.step(actions)

    assert env_terminated is False and env_truncated is True
    assert (not np.any(envs_terminated)) and np.all(envs_truncated)

    actions = envs.action_space.sample()
    env_obs, env_info = env.reset()
    envs_obs, envs_reward, envs_terminated, envs_truncated, envs_info = envs.step(
        actions
    )
    assert np.allclose(env_obs, envs_obs[0])
    assert np.all(envs_reward == np.array([0.0]))
    assert np.all(envs_terminated == np.array([False]))
    assert np.all(envs_truncated == np.array([False]))
    assert data_equivalence(env_info, envs_info)
