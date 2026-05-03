"""A set of functions for checking a wrapper implementation."""

import numpy as np

import gymnasium as gym
from gymnasium import logger
from gymnasium.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
)


def check_wrapper(
    wrapper: gym.Wrapper,
    skip_render_check: bool = False,
):
    """Check that a wrapper follows Gymnasium's Wrapper API.

    Verifies that the wrapper correctly implements the Wrapper interface by checking
    that it has a valid inner environment, observation and action spaces are valid,
    that ``unwrapped`` returns the base environment, and that :meth:`reset` and
    :meth:`step` return data in the correct format with observations contained
    in the declared spaces.

    Args:
        wrapper: The Gymnasium wrapper instance to check.
        skip_render_check: Whether to skip the render method check. ``False`` by default.

    Raises:
        TypeError: If the wrapper does not inherit from :class:`gymnasium.Wrapper`
            or the inner environment is not a :class:`gymnasium.Env`.
        AttributeError: If the wrapper is missing an inner environment,
            observation space, or action space.
        AssertionError: If ``unwrapped``, ``reset``, or ``step`` return invalid data.
    """
    # ============= Check the wrapper type =============
    if not isinstance(wrapper, gym.Wrapper):
        raise TypeError(
            f"The wrapper must inherit from gymnasium.Wrapper, actual class: {type(wrapper)}."
        )

    # ============= Check the inner environment =============
    if not hasattr(wrapper, "env"):
        raise AttributeError(
            "The wrapper must have an inner environment (wrapper.env)."
        )
    if not isinstance(wrapper.env, gym.Env):
        raise TypeError(
            f"The inner environment must be a gymnasium.Env, got {type(wrapper.env)}."
        )

    # ============= Check the spaces (observation and action) =============
    if not hasattr(wrapper, "observation_space"):
        raise AttributeError("The wrapper must specify an observation space.")
    check_observation_space(wrapper.observation_space)

    if not hasattr(wrapper, "action_space"):
        raise AttributeError("The wrapper must specify an action space.")
    check_action_space(wrapper.action_space)

    # ============= Check unwrapped =============
    assert wrapper.unwrapped is not wrapper, (
        "`wrapper.unwrapped` should return the base environment, not the wrapper itself."
    )
    assert isinstance(wrapper.unwrapped, gym.Env), (
        f"`wrapper.unwrapped` must return a gymnasium.Env, got {type(wrapper.unwrapped)}."
    )

    # ============= Check reset() =============
    result = wrapper.reset()
    assert isinstance(result, tuple), f"reset() must return a tuple, got {type(result)}"
    assert len(result) == 2, (
        f"reset() must return 2 values (obs, info), got {len(result)}"
    )

    obs, info = result
    assert obs in wrapper.observation_space, (
        "Observation from reset() is not in observation_space."
    )
    assert isinstance(info, dict), f"info from reset() must be a dict, got {type(info)}"

    # ============= Check step() =============
    action = wrapper.action_space.sample()
    result = wrapper.step(action)
    assert isinstance(result, tuple), f"step() must return a tuple, got {type(result)}"
    assert len(result) == 5, (
        f"step() must return 5 values (obs, reward, terminated, truncated, info), got {len(result)}"
    )

    obs, reward, terminated, truncated, info = result
    assert obs in wrapper.observation_space, (
        "Observation from step() is not in observation_space."
    )

    if not (
        np.issubdtype(type(reward), np.integer)
        or np.issubdtype(type(reward), np.floating)
    ):
        logger.warn(
            f"The reward returned by `step()` must be a float, int, np.integer or np.floating, actual type: {type(reward)}"
        )
    else:
        if np.isnan(reward):
            logger.warn("The reward is a NaN value.")
        if np.isinf(reward):
            logger.warn("The reward is an inf value.")

    assert isinstance(terminated, bool), (
        f"terminated from step() must be a bool, got {type(terminated)}"
    )
    assert isinstance(truncated, bool), (
        f"truncated from step() must be a bool, got {type(truncated)}"
    )
    assert isinstance(info, dict), f"info from step() must be a dict, got {type(info)}"

    # ============= Check render() =============
    if not skip_render_check and wrapper.render_mode is not None:
        wrapper.render()

    # ============= Check close() =============
    wrapper.close()
