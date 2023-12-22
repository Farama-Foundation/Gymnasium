"""A collection of common wrappers.

* ``TimeLimit`` - Provides a time limit on the number of steps for an environment before it truncates
* ``Autoreset`` - Auto-resets the environment
* ``PassiveEnvChecker`` - Passive environment checker that does not modify any environment data
* ``OrderEnforcing`` - Enforces the order of function calls to environments
* ``RecordEpisodeStatistics`` - Records the episode statistics
"""
from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Any, SupportsFloat

import gymnasium as gym
from gymnasium import logger
from gymnasium.core import ActType, ObsType, RenderFrame, WrapperObsType
from gymnasium.error import ResetNeeded
from gymnasium.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)


if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec


__all__ = [
    "TimeLimit",
    "Autoreset",
    "PassiveEnvChecker",
    "OrderEnforcing",
    "RecordEpisodeStatistics",
]


class TimeLimit(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Limits the number of steps for an environment through truncating the environment if a maximum number of timesteps is exceeded.

    If a truncation is not defined inside the environment itself, this is the only place that the truncation signal is issued.
    Critically, this is different from the `terminated` signal that originates from the underlying environment as part of the MDP.
    No vector wrapper exists.

    Example using the TimeLimit wrapper:
        >>> from gymnasium.wrappers import TimeLimit
        >>> from gymnasium.envs.classic_control import CartPoleEnv

        >>> spec = gym.spec("CartPole-v1")
        >>> spec.max_episode_steps
        500
        >>> env = gym.make("CartPole-v1")
        >>> env  # TimeLimit is included within the environment stack
        <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
        >>> env.spec  # doctest: +ELLIPSIS
        EnvSpec(id='CartPole-v1', ..., max_episode_steps=500, ...)
        >>> env = gym.make("CartPole-v1", max_episode_steps=3)
        >>> env.spec  # doctest: +ELLIPSIS
        EnvSpec(id='CartPole-v1', ..., max_episode_steps=3, ...)
        >>> env = TimeLimit(CartPoleEnv(), max_episode_steps=10)
        >>> env
        <TimeLimit<CartPoleEnv instance>>

    Example of `TimeLimit` determining the episode step
        >>> env = gym.make("CartPole-v1", max_episode_steps=3)
        >>> _ = env.reset(seed=123)
        >>> _ = env.action_space.seed(123)
        >>> _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        >>> terminated, truncated
        (False, False)
        >>> _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        >>> terminated, truncated
        (False, False)
        >>> _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        >>> terminated, truncated
        (False, True)

    Change logs:
     * v0.10.6 - Initially added
     * v0.25.0 - With the step API update, the termination and truncation signal is returned separately.
    """

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int,
    ):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: An optional max episode steps (if ``None``, ``env.spec.max_episode_steps`` is used)
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, max_episode_steps=max_episode_steps
        )
        gym.Wrapper.__init__(self, env)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            seed: Seed for the environment
            options: Options for the environment

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return super().reset(seed=seed, options=options)

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to include the `max_episode_steps=self._max_episode_steps`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            env_spec = deepcopy(env_spec)
            env_spec.max_episode_steps = self._max_episode_steps

        self._cached_spec = env_spec
        return env_spec


class Autoreset(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """The wrapped environment is automatically reset when an terminated or truncated state is reached.

    When calling step causes :meth:`Env.step` to return `terminated=True` or `truncated=True`, :meth:`Env.reset` is called,
    and the return format of :meth:`self.step` is as follows: ``(new_obs, final_reward, final_terminated, final_truncated, info)``
    with new step API and ``(new_obs, final_reward, final_done, info)`` with the old step API.
    No vector version of the wrapper exists.

     - ``obs`` is the first observation after calling :meth:`self.env.reset`
     - ``final_reward`` is the reward after calling :meth:`self.env.step`, prior to calling :meth:`self.env.reset`.
     - ``final_terminated`` is the terminated value before calling :meth:`self.env.reset`.
     - ``final_truncated`` is the truncated value before calling :meth:`self.env.reset`. Both `final_terminated` and `final_truncated` cannot be False.
     - ``info`` is a dict containing all the keys from the info dict returned by the call to :meth:`self.env.reset`,
       with an additional key "final_observation" containing the observation returned by the last call to :meth:`self.env.step`
       and "final_info" containing the info dict returned by the last call to :meth:`self.env.step`.

    Warning:
        When using this wrapper to collect rollouts, note that when :meth:`Env.step` returns `terminated` or `truncated`, a
        new observation from after calling :meth:`Env.reset` is returned by :meth:`Env.step` alongside the
        final reward, terminated and truncated state from the previous episode.
        If you need the final state from the previous episode, you need to retrieve it via the
        "final_observation" key in the info dict.
        Make sure you know what you're doing if you use this wrapper!

    Change logs:
     * v0.24.0 - Initially added as `AutoResetWrapper`
     * v1.0.0 - renamed to `Autoreset` and autoreset order was changed to reset on the step after the environment terminates or truncates. As a result, `"final_observation"` and `"final_info"` is removed.
    """

    def __init__(self, env: gym.Env):
        """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`self.step`.

        Args:
            env (gym.Env): The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        self.autoreset = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment and sets autoreset to False preventing."""
        self.autoreset = False
        return super().reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment with action and resets the environment if a terminated or truncated signal is encountered.

        Args:
            action: The action to take

        Returns:
            The autoreset environment :meth:`step`
        """
        if self.autoreset:
            obs, info = self.env.reset()
            reward, terminated, truncated = 0.0, False, False
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)

        self.autoreset = terminated or truncated
        return obs, reward, terminated, truncated, info


class PassiveEnvChecker(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """A passive wrapper that surrounds the ``step``, ``reset`` and ``render`` functions to check they follow Gymnasium's API.

    This wrapper is automatically applied during make and can be disabled with `disable_env_checker`.
    No vector version of the wrapper exists.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> env
        <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
        >>> env = gym.make("CartPole-v1", disable_env_checker=True)
        >>> env
        <TimeLimit<OrderEnforcing<CartPoleEnv<CartPole-v1>>>>

    Change logs:
     * v0.24.1 - Initially added however broken in several ways
     * v0.25.0 - Bugs was all fixed
     * v0.29.0 - Removed warnings for infinite bounds for Box observation and action spaces and inregular bound shapes
    """

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Initialises the wrapper with the environments, run the observation and action space tests."""
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        assert hasattr(
            env, "action_space"
        ), "The environment must specify an action space. https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/"
        check_action_space(env.action_space)
        assert hasattr(
            env, "observation_space"
        ), "The environment must specify an observation space. https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/"
        check_observation_space(env.observation_space)

        self.checked_reset: bool = False
        self.checked_step: bool = False
        self.checked_render: bool = False
        self.close_called: bool = False

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment that on the first call will run the `passive_env_step_check`."""
        if self.checked_step is False:
            self.checked_step = True
            return env_step_passive_checker(self.env, action)
        else:
            return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment that on the first call will run the `passive_env_reset_check`."""
        if self.checked_reset is False:
            self.checked_reset = True
            return env_reset_passive_checker(self.env, seed=seed, options=options)
        else:
            return self.env.reset(seed=seed, options=options)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Renders the environment that on the first call will run the `passive_env_render_check`."""
        if self.checked_render is False:
            self.checked_render = True
            return env_render_passive_checker(self.env)
        else:
            return self.env.render()

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to such that `disable_env_checker=False`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            env_spec = deepcopy(env_spec)
            env_spec.disable_env_checker = False

        self._cached_spec = env_spec
        return env_spec

    def close(self):
        """Warns if calling close on a closed environment fails."""
        if not self.close_called:
            self.close_called = True
            return self.env.close()
        else:
            try:
                return self.env.close()
            except Exception as e:
                logger.warn(
                    "Calling `env.close()` on the closed environment should be allowed, but it raised the following exception."
                )
                raise e


class OrderEnforcing(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Will produce an error if ``step`` or ``render`` is called before ``reset``.

    No vector version of the wrapper exists.

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
        gymnasium.error.ResetNeeded: Cannot call `env.render()` before calling `env.reset()`, if this is an intended action, set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper.
        >>> _ = env.reset()
        >>> env.render()
        >>> _ = env.step(0)
        >>> env.close()

    Change logs:
     * v0.22.0 - Initially added
     * v0.24.0 - Added order enforcing for the render function
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        disable_render_order_enforcing: bool = False,
    ):
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

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        """Steps through the environment."""
        if not self._has_reset:
            raise ResetNeeded("Cannot call env.step() before calling env.reset()")
        return super().step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment with `kwargs`."""
        self._has_reset = True
        return super().reset(seed=seed, options=options)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Renders the environment with `kwargs`."""
        if not self._disable_render_order_enforcing and not self._has_reset:
            raise ResetNeeded(
                "Cannot call `env.render()` before calling `env.reset()`, if this is an intended action, "
                "set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper."
            )
        return super().render()

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


class RecordEpisodeStatistics(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.
    A vector version of the wrapper exists, :class:`gymnasium.wrappers.vector.RecordEpisodeStatistics`.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
     * time_queue: The time length of the last ``deque_size``-many episodes
     * return_queue: The cumulative rewards of the last ``deque_size``-many episodes
     * length_queue: The lengths of the last ``deque_size``-many episodes

    Change logs:
     * v0.15.4 - Initially added
     * v1.0.0 - Removed vector environment support (see :class:`gymnasium.wrappers.vector.RecordEpisodeStatistics`) and add attribute ``time_queue``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        buffer_length: int = 100,
        stats_key: str = "episode",
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            buffer_length: The size of the buffers :attr:`return_queue`, :attr:`length_queue` and :attr:`time_queue`
            stats_key: The info key for the episode statistics
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        self._stats_key = stats_key

        self.episode_count = 0
        self.episode_start_time: float = -1
        self.episode_returns: float = 0.0
        self.episode_lengths: int = 0

        self.time_queue: deque[float] = deque(maxlen=buffer_length)
        self.return_queue: deque[float] = deque(maxlen=buffer_length)
        self.length_queue: deque[int] = deque(maxlen=buffer_length)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, recording the episode statistics."""
        obs, reward, terminated, truncated, info = super().step(action)

        self.episode_returns += reward
        self.episode_lengths += 1

        if terminated or truncated:
            assert self._stats_key not in info

            episode_time_length = round(
                time.perf_counter() - self.episode_start_time, 6
            )
            info[self._stats_key] = {
                "r": self.episode_returns,
                "l": self.episode_lengths,
                "t": episode_time_length,
            }

            self.time_queue.append(episode_time_length)
            self.return_queue.append(self.episode_returns)
            self.length_queue.append(self.episode_lengths)

            self.episode_count += 1
            self.episode_start_time = time.perf_counter()

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment using seed and options and resets the episode rewards and lengths."""
        obs, info = super().reset(seed=seed, options=options)

        self.episode_start_time = time.perf_counter()
        self.episode_returns = 0.0
        self.episode_lengths = 0

        return obs, info
