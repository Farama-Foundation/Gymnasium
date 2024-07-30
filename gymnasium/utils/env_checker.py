"""A set of functions for checking an environment implementation.

This file is originally from the Stable Baselines3 repository hosted on GitHub
(https://github.com/DLR-RM/stable-baselines3/)
Original Author: Antonin Raffin

It also uses some warnings/assertions from the PettingZoo repository hosted on GitHub
(https://github.com/PettingZoo-Team/PettingZoo)
Original Author: J K Terry

This was rewritten and split into "env_checker.py" and "passive_env_checker.py" for invasive and passive environment checking
Original Author: Mark Towers

These projects are covered by the MIT License.
"""

import inspect
from copy import deepcopy

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)


def data_equivalence(data_1, data_2, exact: bool = False) -> bool:
    """Assert equality between data 1 and 2, i.e. observations, actions, info.

    Args:
        data_1: data structure 1
        data_2: data structure 2
        exact: whether to compare array exactly or not if false compares with absolute and relative tolerance of 1e-5 (for more information check [np.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)).

    Returns:
        If observation 1 and 2 are equivalent
    """
    if type(data_1) is not type(data_2):
        return False
    elif isinstance(data_1, dict):
        return data_1.keys() == data_2.keys() and all(
            data_equivalence(data_1[k], data_2[k], exact) for k in data_1.keys()
        )
    elif isinstance(data_1, (tuple, list)):
        return len(data_1) == len(data_2) and all(
            data_equivalence(o_1, o_2, exact) for o_1, o_2 in zip(data_1, data_2)
        )
    elif isinstance(data_1, np.ndarray):
        if data_1.shape == data_2.shape and data_1.dtype == data_2.dtype:
            if data_1.dtype == object:
                return all(
                    data_equivalence(a, b, exact) for a, b in zip(data_1, data_2)
                )
            else:
                if exact:
                    return np.all(data_1 == data_2)
                else:
                    return np.allclose(data_1, data_2, rtol=1e-5, atol=1e-5)
        else:
            return False
    else:
        return data_1 == data_2


def check_reset_seed_determinism(env: gym.Env):
    """Check that the environment can be reset with a seed.

    Args:
        env: The environment to check

    Raises:
        AssertionError: The environment cannot be reset with a random seed,
            even though `seed` or `kwargs` appear in the signature.
    """
    signature = inspect.signature(env.reset)
    if "seed" in signature.parameters or (
        "kwargs" in signature.parameters
        and signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    ):
        try:
            obs_1, info = env.reset(seed=123)
            assert (
                obs_1 in env.observation_space
            ), "The observation returned by `env.reset(seed=123)` is not within the observation space."
            assert (
                env.unwrapped._np_random is not None
            ), "Expects the random number generator to have been generated given a seed was passed to reset. Most likely the environment reset function does not call `super().reset(seed=seed)`."
            seed_123_rng_1 = deepcopy(env.unwrapped._np_random)

            obs_2, info = env.reset()
            assert (
                obs_2 in env.observation_space
            ), "The observation returned by `env.reset()` is not within the observation space."

            obs_3, info = env.reset(seed=123)
            assert (
                obs_3 in env.observation_space
            ), "The observation returned by `env.reset(seed=123)` is not within the observation space."
            seed_123_rng_3 = deepcopy(env.unwrapped._np_random)

            obs_4, info = env.reset()
            assert (
                obs_4 in env.observation_space
            ), "The observation returned by `env.reset()` is not within the observation space."

            if env.spec is not None and env.spec.nondeterministic is False:
                assert data_equivalence(
                    obs_1, obs_3
                ), "Using `env.reset(seed=123)` is non-deterministic as the observations are not equivalent."
                assert data_equivalence(
                    obs_2, obs_4
                ), "Using `env.reset(seed=123)` then `env.reset()` is non-deterministic as the observations are not equivalent."
                if not data_equivalence(obs_1, obs_3, exact=True):
                    logger.warn(
                        "Using `env.reset(seed=123)` observations are not equal although similar."
                    )
                if not data_equivalence(obs_2, obs_4, exact=True):
                    logger.warn(
                        "Using `env.reset(seed=123)` then `env.reset()` observations are not equal although similar."
                    )

            assert (
                seed_123_rng_1.bit_generator.state == seed_123_rng_3.bit_generator.state
            ), "Most likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not same when the same seeds are passed to `env.reset`."

            obs_5, info = env.reset(seed=456)
            assert (
                obs_5 in env.observation_space
            ), "The observation returned by `env.reset(seed=456)` is not within the observation space."
            assert (
                env.unwrapped._np_random.bit_generator.state
                != seed_123_rng_1.bit_generator.state
            ), "Most likely the environment reset function does not call `super().reset(seed=seed)` as the random number generators are not different when different seeds are passed to `env.reset`."

        except TypeError as e:
            raise AssertionError(
                "The environment cannot be reset with a random seed, even though `seed` or `kwargs` appear in the signature. "
                f"This should never happen, please report this issue. The error was: {e}"
            ) from e

        seed_param = signature.parameters.get("seed")
        # Check the default value is None
        if seed_param is not None and seed_param.default is not None:
            logger.warn(
                "The default seed argument in reset should be `None`, otherwise the environment will by default always be deterministic. "
                f"Actual default: {seed_param.default}"
            )
    else:
        raise gym.error.Error(
            "The `reset` method does not provide a `seed` or `**kwargs` keyword argument."
        )


def check_reset_options(env: gym.Env):
    """Check that the environment can be reset with options.

    Args:
        env: The environment to check

    Raises:
        AssertionError: The environment cannot be reset with options,
            even though `options` or `kwargs` appear in the signature.
    """
    signature = inspect.signature(env.reset)
    if "options" in signature.parameters or (
        "kwargs" in signature.parameters
        and signature.parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    ):
        try:
            env.reset(options={})
        except TypeError as e:
            raise AssertionError(
                "The environment cannot be reset with options, even though `options` or `**kwargs` appear in the signature. "
                f"This should never happen, please report this issue. The error was: {e}"
            ) from e
    else:
        raise gym.error.Error(
            "The `reset` method does not provide an `options` or `**kwargs` keyword argument."
        )


def check_step_determinism(env: gym.Env, seed=123):
    """Check that the environment steps deterministically after reset.

    Note: This check assumes that seeded `reset()` is deterministic (it must have passed `check_reset_seed`) and that `step()` returns valid values (passed `env_step_passive_checker`).
    Note: A single step should be enough to assert that the state transition function is deterministic (at least for most environments).

    Raises:
        AssertionError: The environment cannot be step deterministically after resetting with a random seed,
            or it truncates after 1 step.
    """
    if env.spec is not None and env.spec.nondeterministic is True:
        return

    env.action_space.seed(seed)
    action = env.action_space.sample()

    env.reset(seed=seed)
    obs_0, rew_0, term_0, trunc_0, info_0 = env.step(action)
    seeded_rng: np.random.Generator = deepcopy(env.unwrapped._np_random)

    env.reset(seed=seed)
    obs_1, rew_1, term_1, trunc_1, info_1 = env.step(action)

    assert (
        env.unwrapped._np_random.bit_generator.state  # pyright: ignore [reportOptionalMemberAccess]
        == seeded_rng.bit_generator.state
    ), "The `.np_random` is not properly been updated after step."

    assert data_equivalence(
        obs_0, obs_1
    ), "Deterministic step observations are not equivalent for the same seed and action"
    if not data_equivalence(obs_0, obs_1, exact=True):
        logger.warn(
            "Step observations are not equal although similar given the same seed and action"
        )

    assert data_equivalence(
        rew_0, rew_1
    ), "Deterministic step rewards are not equivalent for the same seed and action"
    if not data_equivalence(rew_0, rew_1, exact=True):
        logger.warn(
            "Step rewards are not equal although similar given the same seed and action"
        )

    assert data_equivalence(
        term_0, term_1, exact=True
    ), "Deterministic step termination are not equivalent for the same seed and action"
    assert (
        trunc_0 is False and trunc_1 is False
    ), "Environment truncates after 1 step, something has gone very wrong."

    assert data_equivalence(
        info_0,
        info_1,
    ), "Deterministic step info are not equivalent for the same seed and action"
    if not data_equivalence(info_0, info_1, exact=True):
        logger.warn(
            "Step info are not equal although similar given the same seed and action"
        )


def check_reset_return_info_deprecation(env: gym.Env):
    """Makes sure support for deprecated `return_info` argument is dropped.

    Args:
        env: The environment to check
    Raises:
        UserWarning
    """
    signature = inspect.signature(env.reset)
    if "return_info" in signature.parameters:
        logger.warn(
            "`return_info` is deprecated as an optional argument to `reset`. `reset`"
            "should now always return `obs, info` where `obs` is an observation, and `info` is a dictionary"
            "containing additional information."
        )


def check_seed_deprecation(env: gym.Env):
    """Makes sure support for deprecated function `seed` is dropped.

    Args:
        env: The environment to check
    Raises:
        UserWarning
    """
    seed_fn = getattr(env, "seed", None)
    if callable(seed_fn):
        logger.warn(
            "Official support for the `seed` function is dropped. "
            "Standard practice is to reset gymnasium environments using `env.reset(seed=<desired seed>)`"
        )


def check_reset_return_type(env: gym.Env):
    """Checks that :meth:`reset` correctly returns a tuple of the form `(obs , info)`.

    Args:
        env: The environment to check
    Raises:
        AssertionError depending on spec violation
    """
    result = env.reset()
    assert isinstance(
        result, tuple
    ), f"The result returned by `env.reset()` was not a tuple of the form `(obs, info)`, where `obs` is a observation and `info` is a dictionary containing additional information. Actual type: `{type(result)}`"
    assert (
        len(result) == 2
    ), f"Calling the reset method did not return a 2-tuple, actual length: {len(result)}"

    obs, info = result
    assert (
        obs in env.observation_space
    ), "The first element returned by `env.reset()` is not within the observation space."
    assert isinstance(
        info, dict
    ), f"The second element returned by `env.reset()` was not a dictionary, actual type: {type(info)}"


def check_space_limit(space, space_type: str):
    """Check the space limit for only the Box space as a test that only runs as part of `check_env`."""
    if isinstance(space, spaces.Box):
        if np.any(np.equal(space.low, -np.inf)):
            logger.warn(
                f"A Box {space_type} space minimum value is -infinity. This is probably too low."
            )
        if np.any(np.equal(space.high, np.inf)):
            logger.warn(
                f"A Box {space_type} space maximum value is infinity. This is probably too high."
            )

        # Check that the Box space is normalized
        if space_type == "action":
            if len(space.shape) == 1:  # for vector boxes
                if (
                    np.any(
                        np.logical_and(
                            space.low != np.zeros_like(space.low),
                            np.abs(space.low) != np.abs(space.high),
                        )
                    )
                    or np.any(space.low < -1)
                    or np.any(space.high > 1)
                ):
                    # todo - Add to gymlibrary.ml?
                    logger.warn(
                        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). "
                        "See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information."
                    )
    elif isinstance(space, spaces.Tuple):
        for subspace in space.spaces:
            check_space_limit(subspace, space_type)
    elif isinstance(space, spaces.Dict):
        for subspace in space.values():
            check_space_limit(subspace, space_type)


def check_env(
    env: gym.Env,
    warn: bool = None,
    skip_render_check: bool = False,
    skip_close_check: bool = False,
):
    """Check that an environment follows Gymnasium's API.

    .. py:currentmodule:: gymnasium.Env

    To ensure that an environment is implemented "correctly", ``check_env`` checks that the :attr:`observation_space` and :attr:`action_space` are correct.
    Furthermore, the function will call the :meth:`reset`, :meth:`step` and :meth:`render` functions with a variety of values.

    We highly recommend users call this function after an environment is constructed and within a project's continuous integration to keep an environment update with Gymnasium's API.

    Args:
        env: The Gym environment that will be checked
        warn: Ignored, previously silenced particular warnings
        skip_render_check: Whether to skip the checks for the render method. False by default (useful for the CI)
        skip_close_check: Whether to skip the checks for the close method. False by default
    """
    if warn is not None:
        logger.warn("`check_env(warn=...)` parameter is now ignored.")

    if not isinstance(env, gym.Env):
        if (
            str(env.__class__.__base__) == "<class 'gym.core.Env'>"
            or str(env.__class__.__base__) == "<class 'gym.core.Wrapper'>"
        ):
            raise TypeError(
                "Gym is incompatible with Gymnasium, please update the environment class to `gymnasium.Env`. "
                "See https://gymnasium.farama.org/introduction/create_custom_env/ for more info."
            )
        else:
            raise TypeError(
                f"The environment must inherit from the gymnasium.Env class, actual class: {type(env)}. "
                "See https://gymnasium.farama.org/introduction/create_custom_env/ for more info."
            )
    if env.unwrapped is not env:
        logger.warn(
            f"The environment ({env}) is different from the unwrapped version ({env.unwrapped}). This could effect the environment checker as the environment most likely has a wrapper applied to it. We recommend using the raw environment for `check_env` using `env.unwrapped`."
        )

    if env.metadata.get("jax", False):
        env = gym.wrappers.JaxToNumpy(env)
    elif env.metadata.get("torch", False):
        env = gym.wrappers.TorchToNumpy(env)

    # ============= Check the spaces (observation and action) ================
    if not hasattr(env, "action_space"):
        raise AttributeError(
            "The environment must specify an action space. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info."
        )
    check_action_space(env.action_space)
    check_space_limit(env.action_space, "action")

    if not hasattr(env, "observation_space"):
        raise AttributeError(
            "The environment must specify an observation space. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info."
        )
    check_observation_space(env.observation_space)
    check_space_limit(env.observation_space, "observation")

    # ==== Check the reset method ====
    check_seed_deprecation(env)
    check_reset_return_info_deprecation(env)
    check_reset_return_type(env)
    check_reset_seed_determinism(env)
    check_reset_options(env)

    # ============ Check the returned values ===============
    env_reset_passive_checker(env)
    env_step_passive_checker(env, env.action_space.sample())

    # ==== Check the step method ====
    check_step_determinism(env)

    # ==== Check the render method and the declared render modes ====
    if not skip_render_check:
        if env.render_mode is not None:
            env_render_passive_checker(env)

        if env.spec is not None:
            for render_mode in env.metadata["render_modes"]:
                new_env = env.spec.make(render_mode=render_mode)
                new_env.reset()
                env_render_passive_checker(new_env)
                new_env.close()
        else:
            logger.warn(
                "Not able to test alternative render modes due to the environment not having a spec. Try instantiating the environment through `gymnasium.make`"
            )

    if not skip_close_check and env.spec is not None:
        new_env = env.spec.make()
        new_env.close()
        try:
            new_env.close()
        except Exception as e:
            logger.warn(
                f"Calling `env.close()` on the closed environment should be allowed, but it raised an exception: {e}"
            )
