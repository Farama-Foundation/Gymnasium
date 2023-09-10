import numpy as np

from gymnasium.envs.phys2d.cartpole import CartPoleFunctional
from gymnasium.utils.env_checker import check_env, data_equivalence
from gymnasium.utils.functional_jax_env import FunctionalJaxEnv, FunctionalJaxVectorEnv
from gymnasium.wrappers import TimeLimit


def test_functional_to_env():
    env = FunctionalJaxEnv(CartPoleFunctional())

    check_env(env)


def test_termination_autoreset():
    env = FunctionalJaxEnv(CartPoleFunctional())
    envs = FunctionalJaxVectorEnv(CartPoleFunctional(), num_envs=1)

    envs.action_space.seed(123)
    env.reset(seed=123)
    envs.reset(seed=123)

    env.state = envs.state[0]  # use the same state

    episode_over = False
    while not episode_over:
        actions = envs.action_space.sample()

        env_obs, env_reward, env_terminated, env_truncated, env_info = env.step(actions[0])
        envs_obs, envs_reward, envs_terminated, envs_truncated, envs_info = envs.step(actions)

        assert np.allclose(env_obs, envs_obs[0])
        assert env_reward == envs_reward[0]
        assert env_terminated == envs_terminated[0]
        assert env_truncated == envs_truncated[0]
        assert data_equivalence(env_info, envs_info)

        episode_over = env_terminated or env_truncated

    actions = envs.action_space.sample()
    env_obs, env_info = env.reset()
    envs_obs, envs_reward, envs_terminated, envs_truncated, envs_info = envs.step(actions)
    assert np.all(envs_reward == np.array([0.]))
    assert np.all(envs_terminated == np.array([False]))
    assert np.all(envs_truncated == np.array([False]))
    assert data_equivalence(env_info, envs_info)


def test_truncation_autoreset(time_limit=5):
    env = TimeLimit(FunctionalJaxEnv(CartPoleFunctional()), time_limit)
    envs = FunctionalJaxVectorEnv(CartPoleFunctional(), num_envs=1, max_episode_steps=time_limit)

    envs.action_space.seed(123)
    env.reset(seed=123)
    envs.reset(seed=123)

    env.state = envs.state[0]  # use the same state

    for _ in range(time_limit-1):
        actions = envs.action_space.sample()
        _, _, env_terminated, env_truncated, _ = env.step(actions[0])
        _, _, envs_terminated, envs_truncated, _ = envs.step(actions)

        assert env_terminated is False and env_truncated is False
        assert (not np.any(envs_terminated)) and (not np.any(env_truncated))

    actions = envs.action_space.sample()
    _, _, env_terminated, env_truncated, _ = env.step(actions[0])
    _, _, envs_terminated, envs_truncated, _ = envs.step(actions)

    assert env_terminated is False and env_truncated is True
    assert np.all(envs_terminated == np.array([False]))
    assert np.all(envs_truncated == np.array([True]))

    actions = envs.action_space.sample()
    env_obs, env_info = env.reset()
    envs_obs, envs_reward, envs_terminated, envs_truncated, envs_info = envs.step(actions)
    assert np.all(envs_reward == np.array([0.]))
    assert np.all(envs_terminated == np.array([False]))
    assert np.all(envs_truncated == np.array([False]))
    assert data_equivalence(env_info, envs_info)
