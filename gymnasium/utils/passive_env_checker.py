"""A set of functions for passively checking environment implementations."""

import inspect
from functools import partial
from typing import Callable

import numpy as np

from gymnasium import Space, error, logger, spaces


__all__ = [
    "env_render_passive_checker",
    "env_reset_passive_checker",
    "env_step_passive_checker",
    "check_action_space",
    "check_observation_space",
]


def _check_box_observation_space(observation_space: spaces.Box):
    """Checks that a :class:`Box` observation space is defined in a sensible way.

    Args:
        observation_space: A box observation space
    """
    assert (
        observation_space.low.shape == observation_space.shape
    ), f"The Box observation space shape and low shape have different shapes, low shape: {observation_space.low.shape}, box shape: {observation_space.shape}"
    assert (
        observation_space.high.shape == observation_space.shape
    ), f"The Box observation space shape and high shape have have different shapes, high shape: {observation_space.high.shape}, box shape: {observation_space.shape}"

    if np.any(observation_space.low == observation_space.high):
        logger.warn("A Box observation space maximum and minimum values are equal.")
    elif np.any(observation_space.high < observation_space.low):
        logger.warn("A Box observation space low value is greater than a high value.")


def _check_box_action_space(action_space: spaces.Box):
    """Checks that a :class:`Box` action space is defined in a sensible way.

    Args:
        action_space: A box action space
    """
    assert (
        action_space.low.shape == action_space.shape
    ), f"The Box action space shape and low shape have have different shapes, low shape: {action_space.low.shape}, box shape: {action_space.shape}"
    assert (
        action_space.high.shape == action_space.shape
    ), f"The Box action space shape and high shape have different shapes, high shape: {action_space.high.shape}, box shape: {action_space.shape}"

    if np.any(action_space.low == action_space.high):
        logger.warn("A Box action space maximum and minimum values are equal.")


def check_space(
    space: Space, space_type: str, check_box_space_fn: Callable[[spaces.Box], None]
):
    """A passive check of the environment action space that should not affect the environment."""
    if not isinstance(space, spaces.Space):
        if str(space.__class__.__base__) == "<class 'gym.spaces.space.Space'>":
            raise TypeError(
                f"Gym is incompatible with Gymnasium, please update the environment {space_type}_space to `{str(space.__class__.__base__).replace('gym', 'gymnasium')}`."
            )
        else:
            raise TypeError(
                f"{space_type} space does not inherit from `gymnasium.spaces.Space`, actual type: {type(space)}"
            )

    elif isinstance(space, spaces.Box):
        check_box_space_fn(space)
    elif isinstance(space, spaces.Discrete):
        assert (
            0 < space.n
        ), f"Discrete {space_type} space's number of elements must be positive, actual number of elements: {space.n}"
        assert (
            space.shape == ()
        ), f"Discrete {space_type} space's shape should be empty, actual shape: {space.shape}"
    elif isinstance(space, spaces.MultiDiscrete):
        assert (
            space.shape == space.nvec.shape
        ), f"Multi-discrete {space_type} space's shape must be equal to the nvec shape, space shape: {space.shape}, nvec shape: {space.nvec.shape}"
        assert np.all(
            0 < space.nvec
        ), f"Multi-discrete {space_type} space's all nvec elements must be greater than 0, actual nvec: {space.nvec}"
    elif isinstance(space, spaces.MultiBinary):
        assert np.all(
            0 < np.asarray(space.shape)
        ), f"Multi-binary {space_type} space's all shape elements must be greater than 0, actual shape: {space.shape}"
    elif isinstance(space, spaces.Tuple):
        assert 0 < len(
            space.spaces
        ), f"An empty Tuple {space_type} space is not allowed."
        for subspace in space.spaces:
            check_space(subspace, space_type, check_box_space_fn)
    elif isinstance(space, spaces.Dict):
        assert 0 < len(
            space.spaces.keys()
        ), f"An empty Dict {space_type} space is not allowed."
        for subspace in space.values():
            check_space(subspace, space_type, check_box_space_fn)


check_observation_space = partial(
    check_space,
    space_type="observation",
    check_box_space_fn=_check_box_observation_space,
)
check_action_space = partial(
    check_space, space_type="action", check_box_space_fn=_check_box_action_space
)


def check_obs(obs, observation_space: spaces.Space, method_name: str):
    """Check that the observation returned by the environment correspond to the declared one.

    Args:
        obs: The observation to check
        observation_space: The observation space of the observation
        method_name: The method name that generated the observation
    """
    pre = f"The obs returned by the `{method_name}()` method"
    if isinstance(observation_space, spaces.Discrete):
        if not isinstance(obs, (np.int64, int)):
            logger.warn(f"{pre} should be an int or np.int64, actual type: {type(obs)}")
    elif isinstance(observation_space, spaces.Box):
        if observation_space.shape != ():
            if not isinstance(obs, np.ndarray):
                logger.warn(
                    f"{pre} was expecting a numpy array, actual type: {type(obs)}"
                )
            elif obs.dtype != observation_space.dtype:
                logger.warn(
                    f"{pre} was expecting numpy array dtype to be {observation_space.dtype}, actual type: {obs.dtype}"
                )
    elif isinstance(observation_space, (spaces.MultiBinary, spaces.MultiDiscrete)):
        if not isinstance(obs, np.ndarray):
            logger.warn(f"{pre} was expecting a numpy array, actual type: {type(obs)}")
    elif isinstance(observation_space, spaces.Tuple):
        if not isinstance(obs, tuple):
            logger.warn(f"{pre} was expecting a tuple, actual type: {type(obs)}")
        assert len(obs) == len(
            observation_space.spaces
        ), f"{pre} length is not same as the observation space length, obs length: {len(obs)}, space length: {len(observation_space.spaces)}"
        for sub_obs, sub_space in zip(obs, observation_space.spaces):
            check_obs(sub_obs, sub_space, method_name)
    elif isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict), f"{pre} must be a dict, actual type: {type(obs)}"
        assert (
            obs.keys() == observation_space.spaces.keys()
        ), f"{pre} observation keys is not same as the observation space keys, obs keys: {list(obs.keys())}, space keys: {list(observation_space.spaces.keys())}"
        for space_key in observation_space.spaces.keys():
            check_obs(obs[space_key], observation_space[space_key], method_name)

    try:
        if obs not in observation_space:
            logger.warn(f"{pre} is not within the observation space.")
    except Exception as e:
        logger.warn(f"{pre} is not within the observation space with exception: {e}")


def env_reset_passive_checker(env, **kwargs):
    """A passive check of the `Env.reset` function investigating the returning reset information and returning the data unchanged."""
    signature = inspect.signature(env.reset)
    if "seed" not in signature.parameters and "kwargs" not in signature.parameters:
        logger.deprecation(
            "Current gymnasium version requires that `Env.reset` can be passed a `seed` instead of using `Env.seed` for resetting the environment random number generator."
        )
    else:
        seed_param = signature.parameters.get("seed")
        # Check the default value is None
        if seed_param is not None and seed_param.default is not None:
            logger.warn(
                "The default seed argument in `Env.reset` should be `None`, otherwise the environment will by default always be deterministic. "
                f"Actual default: {seed_param}"
            )

    if "options" not in signature.parameters and "kwargs" not in signature.parameters:
        logger.deprecation(
            "Current gymnasium version requires that `Env.reset` can be passed `options` to allow the environment initialisation to be passed additional information."
        )

    # Checks the result of env.reset with kwargs
    result = env.reset(**kwargs)

    if not isinstance(result, tuple):
        logger.warn(
            f"The result returned by `env.reset()` was not a tuple of the form `(obs, info)`, where `obs` is a observation and `info` is a dictionary containing additional information. Actual type: `{type(result)}`"
        )
    elif len(result) != 2:
        logger.warn(
            "The result returned by `env.reset()` should be `(obs, info)` by default, , where `obs` is a observation and `info` is a dictionary containing additional information."
        )
    else:
        obs, info = result
        check_obs(obs, env.observation_space, "reset")
        assert isinstance(
            info, dict
        ), f"The second element returned by `env.reset()` was not a dictionary, actual type: {type(info)}"
    return result


def env_step_passive_checker(env, action):
    """A passive check for the environment step, investigating the returning data then returning the data unchanged."""
    # We don't check the action as for some environments then out-of-bounds values can be given
    result = env.step(action)
    assert isinstance(
        result, tuple
    ), f"Expects step result to be a tuple, actual type: {type(result)}"
    if len(result) == 4:
        logger.deprecation(
            "Core environment is written in old step API which returns one bool instead of two. "
            "It is recommended to rewrite the environment with new step API. "
        )
        obs, reward, done, info = result

        if not isinstance(done, (bool, np.bool_)):
            logger.warn(
                f"Expects `done` signal to be a boolean, actual type: {type(done)}"
            )
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result

        # np.bool is actual python bool not np boolean type, therefore bool_ or bool8
        if not isinstance(terminated, (bool, np.bool_)):
            logger.warn(
                f"Expects `terminated` signal to be a boolean, actual type: {type(terminated)}"
            )
        if not isinstance(truncated, (bool, np.bool_)):
            logger.warn(
                f"Expects `truncated` signal to be a boolean, actual type: {type(truncated)}"
            )
    else:
        raise error.Error(
            f"Expected `Env.step` to return a four or five element tuple, actual number of elements returned: {len(result)}."
        )

    check_obs(obs, env.observation_space, "step")

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

    assert isinstance(
        info, dict
    ), f"The `info` returned by `step()` must be a python dictionary, actual type: {type(info)}"

    return result


def _check_render_return(render_mode, render_return):
    """Produces warning if `render_return` doesn't match `render_mode`."""
    if render_mode == "human":
        if render_return is not None:
            logger.warn(
                f"Human rendering should return `None`, got {type(render_return)}"
            )
    elif render_mode == "rgb_array":
        if not isinstance(render_return, np.ndarray):
            logger.warn(
                f"RGB-array rendering should return a numpy array, got {type(render_return)}"
            )
        else:
            if render_return.dtype != np.uint8:
                logger.warn(
                    f"RGB-array rendering should return a numpy array with dtype uint8, got {render_return.dtype}"
                )
            if render_return.ndim != 3:
                logger.warn(
                    f"RGB-array rendering should return a numpy array with three axes, got {render_return.ndim}"
                )
            if render_return.ndim == 3 and render_return.shape[2] != 3:
                logger.warn(
                    f"RGB-array rendering should return a numpy array in which the last axis has three dimensions, got {render_return.shape[2]}"
                )
    elif render_mode == "depth_array":
        if not isinstance(render_return, np.ndarray):
            logger.warn(
                f"Depth-array rendering should return a numpy array, got {type(render_return)}"
            )
        elif render_return.ndim != 2:
            logger.warn(
                f"Depth-array rendering should return a numpy array with two axes, got {render_return.ndim}"
            )
    elif render_mode in ["ansi", "ascii"]:
        if not isinstance(render_return, str):
            logger.warn(
                f"ANSI/ASCII rendering should produce a string, got {type(render_return)}"
            )
    elif render_mode.endswith("_list"):
        if not isinstance(render_return, list):
            logger.warn(
                f"Render mode `{render_mode}` should produce a list, got {type(render_return)}"
            )
        else:
            base_render_mode = render_mode[: -len("_list")]
            for item in render_return:
                _check_render_return(
                    base_render_mode, item
                )  # Check that each item of the list matches the base render mode


def env_render_passive_checker(env):
    """A passive check of the `Env.render` that the declared render modes/fps in the metadata of the environment is declared."""
    render_modes = env.metadata.get("render_modes")
    if render_modes is None:
        logger.warn(
            "No render modes was declared in the environment (env.metadata['render_modes'] is None or not defined), you may have trouble when calling `.render()`."
        )
    else:
        if not isinstance(render_modes, (list, tuple)):
            logger.warn(
                f"Expects the render_modes to be a sequence (i.e. list, tuple), actual type: {type(render_modes)}"
            )
        elif not all(isinstance(mode, str) for mode in render_modes):
            logger.warn(
                f"Expects all render modes to be strings, actual types: {[type(mode) for mode in render_modes]}"
            )

        render_fps = env.metadata.get("render_fps")
        # We only require `render_fps` if rendering is actually implemented
        if len(render_modes) > 0:
            if render_fps is None:
                logger.warn(
                    "No render fps was declared in the environment (env.metadata['render_fps'] is None or not defined), rendering may occur at inconsistent fps."
                )
            else:
                if not (
                    np.issubdtype(type(render_fps), np.integer)
                    or np.issubdtype(type(render_fps), np.floating)
                ):
                    logger.warn(
                        f"Expects the `env.metadata['render_fps']` to be an integer or a float, actual type: {type(render_fps)}"
                    )
                else:
                    assert (
                        render_fps > 0
                    ), f"Expects the `env.metadata['render_fps']` to be greater than zero, actual value: {render_fps}"

        # env.render is now an attribute with default None
        if len(render_modes) == 0:
            assert (
                env.render_mode is None
            ), f"With no render_modes, expects the Env.render_mode to be None, actual value: {env.render_mode}"
        else:
            assert env.render_mode is None or env.render_mode in render_modes, (
                "The environment was initialized successfully however with an unsupported render mode. "
                f"Render mode: {env.render_mode}, modes: {render_modes}"
            )

    result = env.render()
    if env.render_mode is not None:
        _check_render_return(env.render_mode, result)

    return result
