"""Wrapper to enforce the proper ordering of environment operations."""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import gymnasium as gym
from gymnasium.error import ResetNeeded


if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec


class OrderEnforcing(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """A wrapper that will produce an error if :meth:`step` is called before an initial :meth:`reset`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import OrderEnforcing
        >>> env = gym.make("CartPole-v1", render_mode="human")
        >>> env = OrderEnforcing(env)
        >>> env.step(0)
        Traceback (most recent call last):
            ...
        gymnasium.error.ResetNeeded: Cannot call env.step() before calling env.reset()
        >>> env.render()
        Traceback (most recent call last):
            ...
        gymnasium.error.ResetNeeded: Cannot call `env.render()` before calling `env.reset()`, if this is a intended action, set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper.
        >>> _ = env.reset()
        >>> env.render()
        >>> _ = env.step(0)
        >>> env.close()
    """

    def __init__(self, env: gym.Env, disable_render_order_enforcing: bool = False):
        """A wrapper that will produce an error if :meth:`step` is called before an initial :meth:`reset`.

        Args:
            env: The environment to wrap
            disable_render_order_enforcing: If to disable render order enforcing
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, disable_render_order_enforcing=disable_render_order_enforcing
        )
        gym.Wrapper.__init__(self, env)

        self._has_reset: bool = False
        self._disable_render_order_enforcing: bool = disable_render_order_enforcing

    def step(self, action):
        """Steps through the environment with `kwargs`."""
        if not self._has_reset:
            raise ResetNeeded("Cannot call env.step() before calling env.reset()")
        return self.env.step(action)

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""
        self._has_reset = True
        return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        """Renders the environment with `kwargs`."""
        if not self._disable_render_order_enforcing and not self._has_reset:
            raise ResetNeeded(
                "Cannot call `env.render()` before calling `env.reset()`, if this is a intended action, "
                "set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper."
            )
        return self.env.render(*args, **kwargs)

    @property
    def has_reset(self):
        """Returns if the environment has been reset before."""
        return self._has_reset

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to add the `order_enforce=True`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            env_spec = deepcopy(env_spec)
            env_spec.order_enforce = True

        self._cached_spec = env_spec
        return env_spec
