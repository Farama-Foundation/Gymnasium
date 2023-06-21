"""A passive environment checker wrapper for an environment's observation and action space along with the reset, step and render functions."""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import gymnasium as gym
from gymnasium import logger
from gymnasium.core import ActType
from gymnasium.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)


if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec


class PassiveEnvChecker(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """A passive environment checker wrapper that surrounds the step, reset and render functions to check they follow the gymnasium API."""

    def __init__(self, env):
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

        self.checked_reset = False
        self.checked_step = False
        self.checked_render = False
        self.close_called = False

    def step(self, action: ActType):
        """Steps through the environment that on the first call will run the `passive_env_step_check`."""
        if not self.checked_step:
            self.checked_step = True
            return env_step_passive_checker(self.env, action)
        else:
            return self.env.step(action)

    def reset(self, **kwargs):
        """Resets the environment that on the first call will run the `passive_env_reset_check`."""
        if not self.checked_reset:
            self.checked_reset = True
            return env_reset_passive_checker(self.env, **kwargs)
        else:
            return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        """Renders the environment that on the first call will run the `passive_env_render_check`."""
        if not self.checked_render:
            self.checked_render = True
            return env_render_passive_checker(self.env, *args, **kwargs)
        else:
            return self.env.render(*args, **kwargs)

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
