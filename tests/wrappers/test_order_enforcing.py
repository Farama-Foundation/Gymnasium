"""Test suite for OrderEnforcing wrapper."""

import pytest

from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.error import ResetNeeded
from gymnasium.wrappers import OrderEnforcing
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
