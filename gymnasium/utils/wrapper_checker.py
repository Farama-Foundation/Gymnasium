"""Utility for checking that a Gymnasium wrapper is correctly implemented."""
from __future__ import annotations

import warnings

import gymnasium as gym
from gymnasium.utils.passive_env_checker import check_observation_space, check_action_space


def check_wrapper(env: gym.Wrapper, skip_render_check: bool = True) -> None:
    """Check that a wrapper is correctly implemented.

    This is a sanity check that runs a few steps and verifies the wrapper
    respects the Gymnasium API contracts: spaces are valid, observations
    returned by ``reset`` and ``step`` are contained in ``observation_space``,
    and actions sampled from ``action_space`` are accepted by ``step``.

    Args:
        env: The wrapped environment to check. Must be an instance of
            :class:`gymnasium.Wrapper`.
        skip_render_check: Whether to skip the render check. Defaults to
            ``True`` since rendering is optional and may require a display.

    Raises:
        AssertionError: If any of the checks fail.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.utils.wrapper_checker import check_wrapper
        >>> env = gym.make("CartPole-v1")
        >>> env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        >>> check_wrapper(env)
    """
    assert isinstance(env, gym.Wrapper), (
        f"check_wrapper expects a gymnasium.Wrapper, got {type(env)}"
    )

    # --- spaces ---
    assert hasattr(env, "observation_space"), (
        "Wrapper is missing `observation_space`."
    )
    assert hasattr(env, "action_space"), (
        "Wrapper is missing `action_space`."
    )

    check_observation_space(env.observation_space)
    check_action_space(env.action_space)

    # --- reset ---
    result = env.reset()
    assert isinstance(result, tuple) and len(result) == 2, (
        f"`reset()` must return (obs, info), got {type(result)}"
    )
    obs, info = result
    assert isinstance(info, dict), (
        f"`reset()` info must be a dict, got {type(info)}"
    )
    assert env.observation_space.contains(obs), (
        f"Observation returned by `reset()` is not in `observation_space`.\n"
        f"obs={obs}\nobservation_space={env.observation_space}"
    )

    # --- step ---
    action = env.action_space.sample()
    result = env.step(action)
    assert isinstance(result, tuple) and len(result) == 5, (
        f"`step()` must return (obs, reward, terminated, truncated, info), got {type(result)}"
    )
    obs, reward, terminated, truncated, info = result

    assert env.observation_space.contains(obs), (
        f"Observation returned by `step()` is not in `observation_space`.\n"
        f"obs={obs}\nobservation_space={env.observation_space}"
    )
    assert isinstance(reward, (int, float)), (
        f"Reward must be a scalar, got {type(reward)}"
    )
    assert isinstance(terminated, bool), (
        f"`terminated` must be bool, got {type(terminated)}"
    )
    assert isinstance(truncated, bool), (
        f"`truncated` must be bool, got {type(truncated)}"
    )
    assert isinstance(info, dict), (
        f"`step()` info must be a dict, got {type(info)}"
    )

    # --- unwrapped env accessible ---
    assert hasattr(env, "unwrapped"), "Wrapper is missing `unwrapped` attribute."
    assert isinstance(env.unwrapped, gym.Env), (
        f"`unwrapped` must be a gym.Env, got {type(env.unwrapped)}"
    )

    # --- warn if spaces changed but wrapper doesn't document it ---
    if env.observation_space != env.unwrapped.observation_space:
        warnings.warn(
            "Wrapper changes `observation_space` relative to the unwrapped env. "
            "Make sure `observation_method` is overridden accordingly.",
            UserWarning,
            stacklevel=2,
        )
    if env.action_space != env.unwrapped.action_space:
        warnings.warn(
            "Wrapper changes `action_space` relative to the unwrapped env. "
            "Make sure `action` is overridden accordingly.",
            UserWarning,
            stacklevel=2,
        )

    env.close()
