"""A collection of common wrappers.

* ``AutoresetV0`` - Auto-resets the environment
* ``PassiveEnvCheckerV0`` - Passive environment checker that does not modify any environment data
* ``OrderEnforcingV0`` - Enforces the order of function calls to environments
* ``RecordEpisodeStatisticsV0`` - Records the episode statistics
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium import Env
from gymnasium.core import ActType, ObsType, RenderFrame, WrapperActType, WrapperObsType
from gymnasium.error import ResetNeeded
from gymnasium.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)


class AutoresetV0(gym.Wrapper):
    """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`self.step`."""

    def __init__(self, env: gym.Env):
        """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`self.step`.

        Args:
            env (gym.Env): The environment to apply the wrapper
        """
        super().__init__(env)
        self._episode_ended: bool = False
        self._reset_options: dict[str, Any] | None = None

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict]:
        """Steps through the environment with action and resets the environment if a terminated or truncated signal is encountered in the previous step.

        Args:
            action: The action to take

        Returns:
            The autoreset environment :meth:`step`
        """
        if self._episode_ended:
            obs, info = super().reset(options=self._reset_options)
            self._episode_ended = True
            return obs, 0, False, False, info
        else:
            obs, reward, terminated, truncated, info = super().step(action)
            self._episode_ended = terminated or truncated
            return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment, saving the options used."""
        self._episode_ended = False
        self._reset_options = options
        return super().reset(seed=seed, options=self._reset_options)


class PassiveEnvCheckerV0(gym.Wrapper):
    """A passive environment checker wrapper that surrounds the step, reset and render functions to check they follow the gymnasium API."""

    def __init__(self, env: Env[ObsType, ActType]):
        """Initialises the wrapper with the environments, run the observation and action space tests."""
        super().__init__(env)

        assert hasattr(
            env, "action_space"
        ), "The environment must specify an action space. https://gymnasium.farama.org/content/environment_creation/"
        check_action_space(env.action_space)
        assert hasattr(
            env, "observation_space"
        ), "The environment must specify an observation space. https://gymnasium.farama.org/content/environment_creation/"
        check_observation_space(env.observation_space)

        self._checked_reset: bool = False
        self._checked_step: bool = False
        self._checked_render: bool = False

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment that on the first call will run the `passive_env_step_check`."""
        if self._checked_step is False:
            self._checked_step = True
            return env_step_passive_checker(self.env, action)
        else:
            return self.env.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment that on the first call will run the `passive_env_reset_check`."""
        if self._checked_reset is False:
            self._checked_reset = True
            return env_reset_passive_checker(self.env, seed=seed, options=options)
        else:
            return self.env.reset(seed=seed, options=options)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Renders the environment that on the first call will run the `passive_env_render_check`."""
        if self._checked_render is False:
            self._checked_render = True
            return env_render_passive_checker(self.env)
        else:
            return self.env.render()


class OrderEnforcingV0(gym.Wrapper):
    """A wrapper that will produce an error if :meth:`step` is called before an initial :meth:`reset`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.experimental.wrappers import OrderEnforcingV0
        >>> env = gym.make("CartPole-v1", render_mode="human")
        >>> env = OrderEnforcingV0(env)
        >>> env.step(0) # doctest: +SKIP
        gymnasium.error.ResetNeeded: Cannot call env.step() before calling env.reset()
        >>> env.render() # doctest: +SKIP
        gymnasium.error.ResetNeeded('Cannot call `env.render()` before calling `env.reset()`, if this is a intended action, set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper.')
        >>> _ = env.reset()
        >>> env.render()
        >>> _ = env.step(0)
    """

    def __init__(self, env: gym.Env, disable_render_order_enforcing: bool = False):
        """A wrapper that will produce an error if :meth:`step` is called before an initial :meth:`reset`.

        Args:
            env: The environment to wrap
            disable_render_order_enforcing: If to disable render order enforcing
        """
        super().__init__(env)
        self._has_reset: bool = False
        self._disable_render_order_enforcing: bool = disable_render_order_enforcing

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict]:
        """Steps through the environment with `kwargs`."""
        if not self._has_reset:
            raise ResetNeeded("Cannot call env.step() before calling env.reset()")
        return super().step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment with `kwargs`."""
        self._has_reset = True
        return super().reset(seed=seed, options=options)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Renders the environment with `kwargs`."""
        if not self._disable_render_order_enforcing and not self._has_reset:
            raise ResetNeeded(
                "Cannot call `env.render()` before calling `env.reset()`, if this is a intended action, "
                "set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper."
            )
        return super().render()

    @property
    def has_reset(self):
        """Returns if the environment has been reset before."""
        return self._has_reset


class RecordEpisodeStatisticsV0(gym.Wrapper):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

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
        episode_reward_buffer: The cumulative rewards of the last ``deque_size``-many episodes
        episode_length_buffer: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        buffer_length: int | None = 100,
        stats_key: str = "episode",
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            buffer_length: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
            stats_key: The info key for the episode statistics
        """
        super().__init__(env)

        self._stats_key = stats_key

        self.episode_count = 0
        self.episode_start_time: float = -1
        self.episode_reward: float = -1
        self.episode_length: int = -1

        self.episode_time_length_buffer = deque(maxlen=buffer_length)
        self.episode_reward_buffer = deque(maxlen=buffer_length)
        self.episode_length_buffer = deque(maxlen=buffer_length)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, recording the episode statistics."""
        obs, reward, terminated, truncated, info = super().step(action)

        self.episode_reward += reward
        self.episode_length += 1

        if terminated or truncated:
            assert self._stats_key not in info

            episode_time_length = np.round(
                time.perf_counter() - self.episode_start_time, 6
            )
            info[self._stats_key] = {
                "r": self.episode_reward,
                "l": self.episode_length,
                "t": episode_time_length,
            }

            self.episode_time_length_buffer.append(episode_time_length)
            self.episode_reward_buffer.append(self.episode_reward)
            self.episode_length_buffer.append(self.episode_length)

            self.episode_count += 1

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment using seed and options and resets the episode rewards and lengths."""
        obs, info = super().reset(seed=seed, options=options)

        self.episode_start_time = time.perf_counter()
        self.episode_reward = 0
        self.episode_length = 0

        return obs, info
