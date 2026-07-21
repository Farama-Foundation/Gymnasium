import pytest
import numpy as np
import jax
# Enable 64-bit precision for JAX to match MuJoCo
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import gymnasium as gym
from mujoco import mjx

@pytest.mark.parametrize("env_name", [
    "Ant-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Humanoid-v5",
    "HumanoidStandup-v5",
    "InvertedDoublePendulum-v5",
    "InvertedPendulum-v5",
    "Pusher-v5",
    "Reacher-v5",
    "Swimmer-v5",
    "Walker2d-v5",
])
def test_mjx_mujoco_similar(env_name):
    try:
        gym.spec(env_name)
        gym.spec("MJX/" + env_name)
    except Exception:
        pytest.skip(f"{env_name} or MJX/{env_name} not registered")

    env_a = gym.make(env_name)
    env_b = gym.make("MJX/" + env_name)

    assert env_a.unwrapped.model.opt.integrator == env_b.unwrapped.func_env.model.opt.integrator, "Integrator mismatch"
    assert env_a.unwrapped.model.opt.iterations == env_b.unwrapped.func_env.model.opt.iterations, "Iterations mismatch"
    assert env_a.unwrapped.model.opt.solver == env_b.unwrapped.func_env.model.opt.solver, "Solver mismatch"

    try:
        env_a.action_space.seed(0)
        obs_a, info_a = env_a.reset(seed=0)
        obs_b, info_b = env_b.reset(seed=0)

        # Copy state from env A to MJX env B
        qpos = env_a.unwrapped.data.qpos
        qvel = env_a.unwrapped.data.qvel
        env_b.unwrapped.state = env_b.unwrapped.state.replace(qpos=jnp.array(qpos), qvel=jnp.array(qvel))
        env_b.unwrapped.state = mjx.forward(env_b.unwrapped.func_env.mjx_model, env_b.unwrapped.state)

        for step in range(100):
            #print(f"Step {step}")
            action = env_a.action_space.sample()
            obs_a, rew_a, term_a, trunc_a, info_a = env_a.step(action)
            obs_b, rew_b, term_b, trunc_b, info_b = env_b.step(jnp.array(action))
            #print(f"rew_a: {rew_a}, rew_b: {rew_b}")

            np.testing.assert_allclose(env_a.unwrapped.data.qpos, env_b.unwrapped.state.qpos, atol=1e-3, rtol=1e-6)
            np.testing.assert_allclose(env_a.unwrapped.data.qvel, env_b.unwrapped.state.qvel, atol=1e-3, rtol=1e-6)
            np.testing.assert_allclose(obs_a, obs_b, atol=1e-3, rtol=1e-6)
            np.testing.assert_allclose(rew_a, rew_b, atol=1e-3, rtol=1e-6)
            #assert term_a == term_b
            #assert trunc_a == trunc_b

    finally:
        env_a.close()
        env_b.close()
