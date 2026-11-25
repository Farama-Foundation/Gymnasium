import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import jax.random as jrng  # noqa: E402
import numpy as np  # noqa: E402

from gymnasium.envs.phys2d.cartpole import (  # noqa: E402
    CartPoleFunctional,
    CartPoleJaxVectorEnv,
)
from gymnasium.envs.phys2d.pendulum import (  # noqa: E402
    PendulumFunctional,
    PendulumJaxVectorEnv,
)


@pytest.mark.parametrize("env_class", [CartPoleFunctional, PendulumFunctional])
def test_normal(env_class):
    env = env_class()
    rng = jrng.PRNGKey(0)

    state = env.initial(rng)
    env.action_space.seed(0)

    for _ in range(10):
        obs = env.observation(state, rng)
        action = env.action_space.sample()
        next_state = env.transition(state, action, rng)
        reward = env.reward(state, action, next_state, rng)
        terminal = env.terminal(next_state, rng)

        assert next_state.shape == state.shape
        try:
            float(reward)
        except ValueError:
            pytest.fail("Reward is not castable to float")
        try:
            bool(terminal)
        except ValueError:
            pytest.fail("Terminal is not castable to bool")

        assert next_state.dtype == jnp.float32
        assert isinstance(obs, jax.Array)
        assert obs.dtype == jnp.float32

        state = next_state


@pytest.mark.parametrize("env_class", [CartPoleFunctional, PendulumFunctional])
def test_jit(env_class):
    env = env_class()
    rng = jrng.PRNGKey(0)

    env.transform(jax.jit)
    state = env.initial(rng)
    env.action_space.seed(0)

    for _ in range(10):
        obs = env.observation(state, rng)
        action = env.action_space.sample()
        next_state = env.transition(state, action, rng)
        reward = env.reward(state, action, next_state, rng)
        terminal = env.terminal(next_state, rng)

        assert next_state.shape == state.shape
        try:
            float(reward)
        except ValueError:
            pytest.fail("Reward is not castable to float")
        try:
            bool(terminal)
        except ValueError:
            pytest.fail("Terminal is not castable to bool")

        assert next_state.dtype == jnp.float32
        assert isinstance(obs, jax.Array)
        assert obs.dtype == jnp.float32

        state = next_state


@pytest.mark.parametrize("env_class", [CartPoleFunctional, PendulumFunctional])
def test_vmap(env_class):
    env = env_class()
    num_envs = 10
    rng = jrng.split(jrng.PRNGKey(0), num_envs)

    env.transform(jax.vmap)
    env.transform(jax.jit)
    state = env.initial(rng)
    env.action_space.seed(0)

    for _ in range(10):
        obs = env.observation(state, None)
        action = jnp.array([env.action_space.sample() for _ in range(num_envs)])
        # if isinstance(env.action_space, Discrete):
        #     action = action.reshape((num_envs, 1))
        next_state = env.transition(state, action, None)
        terminal = env.terminal(next_state, None)
        reward = env.reward(state, action, next_state, None)

        assert next_state.shape == state.shape
        assert next_state.dtype == jnp.float32
        assert reward.shape == (num_envs,)
        assert reward.dtype == jnp.float32
        assert terminal.shape == (num_envs,)
        assert terminal.dtype == np.bool_
        assert isinstance(obs, jax.Array)
        assert obs.dtype == jnp.float32

        state = next_state


@pytest.mark.parametrize("env_class", [CartPoleJaxVectorEnv, PendulumJaxVectorEnv])
def test_vectorized(env_class):
    env = env_class(num_envs=10)
    env.action_space.seed(0)

    obs, info = env.reset(seed=0)
    assert obs.shape == (10,) + env.single_observation_space.shape
    assert isinstance(obs, jax.Array)
    assert isinstance(info, dict)

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (10,) + env.single_observation_space.shape
        assert isinstance(obs, jax.Array)
        assert reward.shape == (10,)
        assert isinstance(reward, jax.Array)
        assert terminated.shape == (10,)
        assert isinstance(terminated, jax.Array)
        assert truncated.shape == (10,)
        assert isinstance(truncated, jax.Array)
        assert isinstance(info, dict)

        # These were removed in the new autoreset order
        assert "final_observation" not in info
        assert "final_info" not in info
        assert "_final_observation" not in info
        assert "_final_info" not in info


def test_next_step_autoreset_reward():
    """After a sub-environment's episode ends, the next step resets it and
    must return a reward of 0, as documented for AutoresetMode.NEXT_STEP."""
    env = CartPoleJaxVectorEnv(num_envs=1)
    env.reset(seed=0)

    terminated = False
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(jnp.array([0]))
        if terminated[0]:
            break
    assert terminated[0], "expected the episode to terminate"

    obs, reward, terminated, truncated, info = env.step(jnp.array([1]))
    assert reward[0] == 0.0
    assert not terminated[0] and not truncated[0]


def test_reset_clears_pending_autoreset():
    """An explicit reset() must clear a pending autoreset so the following
    step is a real transition, not a discarded reset step."""
    env = CartPoleJaxVectorEnv(num_envs=1)
    env.reset(seed=0)

    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(jnp.array([0]))
        if terminated[0]:
            break
    assert terminated[0], "expected the episode to terminate"

    env.reset(seed=123)
    env.step(jnp.array([1]))
    assert env.steps[0] == 1


def _mirror_step(func_env, num_envs, rng, state, action, did_reset, reset_count):
    """Recompute the exact expected outputs of FunctionalJaxVectorEnv.step
    from the FuncEnv primitives and the env's internal rng stream."""
    trans_rng, rng = jrng.split(rng)
    trans_rng = jrng.split(trans_rng, num_envs)

    next_state = func_env.transition(state, action, trans_rng)
    reward = func_env.reward(state, action, next_state, trans_rng)
    if did_reset:
        _, rng = jrng.split(rng)
    obs_rng = jrng.split(rng, num_envs)
    return next_state, reward, obs_rng


def test_mixed_batch_autoreset_exact_transitions():
    """During a mixed-batch autoreset, active lanes must keep their exact
    forward transition, reward and observation, and reset lanes must
    restart with reward 0. Covers termination- and truncation-driven
    staggered autoresets."""
    num_envs = 4
    env = CartPoleJaxVectorEnv(num_envs=num_envs, max_episode_steps=40)
    env.reset(seed=0)

    was_term = np.zeros(num_envs, dtype=bool)
    was_trunc = np.zeros(num_envs, dtype=bool)
    checked_termination_reset = False
    checked_truncation_reset = False

    for _ in range(200):
        # lane 0 balances, the rest push left: staggered terminations and a
        # staggered truncation when lane 0 reaches the time limit
        theta = np.asarray(env.state)[:, 2]
        action = np.zeros(num_envs, dtype=np.int64)
        action[0] = 1 if theta[0] > 0 else 0

        prev_done = was_term | was_trunc
        reset_lanes = np.where(prev_done)[0]
        mixed_reset = 0 < len(reset_lanes) < num_envs
        prev_state, prev_rng = env.state, env.rng

        obs, reward, terminated, truncated, info = env.step(jnp.array(action))

        if mixed_reset:
            active = ~prev_done
            exp_next, exp_reward, exp_obs_rng = _mirror_step(
                env.func_env,
                num_envs,
                prev_rng,
                prev_state,
                jnp.array(action),
                did_reset=True,
                reset_count=len(reset_lanes),
            )
            # active lanes: exact forward transition, reward and observation
            np.testing.assert_array_equal(
                np.asarray(env.state)[active], np.asarray(exp_next)[active]
            )
            np.testing.assert_array_equal(
                np.asarray(reward)[active], np.asarray(exp_reward)[active]
            )
            exp_obs = env.func_env.observation(env.state, exp_obs_rng)
            np.testing.assert_array_equal(np.asarray(obs), np.asarray(exp_obs))
            # reset lanes: fresh start, no reward, no done flags
            assert np.all(np.asarray(reward)[prev_done] == 0.0)
            assert not np.any(np.asarray(terminated)[prev_done])
            assert not np.any(np.asarray(truncated)[prev_done])
            assert np.all(np.asarray(env.steps)[prev_done] == 0)
            checked_termination_reset |= bool(np.any(was_term[prev_done]))
            checked_truncation_reset |= bool(np.any(was_trunc[prev_done]))

        was_term = np.asarray(terminated)
        was_trunc = np.asarray(truncated)
        if checked_termination_reset and checked_truncation_reset:
            break

    assert checked_termination_reset, "no staggered termination autoreset checked"
    assert checked_truncation_reset, "no staggered truncation autoreset checked"
