"""Test suite for OrderEnforcing wrapper."""

import pytest

from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.error import Error, ResetNeeded
from gymnasium.wrappers import OrderEnforcing
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import has_wrapper


def test_order_enforcing():
    """Checks that the order enforcing works as expected, raising an error before reset is called and not after."""
    # The reason for not using gym.make is that all environments are by default wrapped in the order enforcing wrapper
    env = CartPoleEnv(render_mode="rgb_array_list")
    assert not has_wrapper(env, OrderEnforcing)

    # Assert that the order enforcing works for step and render before reset
    order_enforced_env = OrderEnforcing(env)
    assert order_enforced_env.has_reset is False
    with pytest.raises(ResetNeeded):
        order_enforced_env.step(0)
    with pytest.raises(ResetNeeded):
        order_enforced_env.render()
    assert order_enforced_env.has_reset is False

    # Assert that the Assertion errors are not raised after reset
    order_enforced_env.reset()
    assert order_enforced_env.has_reset is True
    order_enforced_env.step(0)
    order_enforced_env.render()

    # Assert that with disable_render_order_enforcing works, the environment has already been reset
    env = CartPoleEnv(render_mode="rgb_array_list")
    env = OrderEnforcing(env, disable_render_order_enforcing=True)
    env.render()  # no assertion error


@pytest.mark.parametrize("disable_render_order_enforcing", [False, True])
def test_failed_initial_reset(disable_render_order_enforcing):
    """A failed first reset must keep order enforcement active until reset succeeds."""
    env = OrderEnforcing(
        GenericTestEnv(),
        disable_render_order_enforcing=disable_render_order_enforcing,
    )
    with pytest.raises(Error, match="Seed must be greater or equal to zero"):
        env.reset(seed=-1)

    assert env.has_reset is False
    with pytest.raises(ResetNeeded):
        env.step(0)
    if disable_render_order_enforcing:
        assert env.render() is None
    else:
        with pytest.raises(ResetNeeded):
            env.render()

    options = {"test": "reset options"}
    obs, info = env.reset(seed=123, options=options)
    assert env.has_reset is True
    assert obs in env.observation_space
    assert info["options"] is options
    env.step(0)
    assert env.render() is None
    env.close()


def test_failed_reset_after_successful_reset():
    """A later reset failure must not erase the record of a successful reset."""
    env = OrderEnforcing(GenericTestEnv())
    env.reset(seed=123)
    with pytest.raises(Error, match="Seed must be greater or equal to zero"):
        env.reset(seed=-1)

    assert env.has_reset is True
    env.step(0)
    assert env.render() is None
    env.close()
