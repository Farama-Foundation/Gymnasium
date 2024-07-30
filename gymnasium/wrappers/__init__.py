"""Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly.

Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular.
Importantly wrappers can be chained to combine their effects and most environments that are generated via
:meth:`gymnasium.make` will already be wrapped by default.

In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along
with (possibly optional) parameters to the wrapper's constructor.

    >>> import gymnasium as gym
    >>> from gymnasium.wrappers import RescaleAction
    >>> base_env = gym.make("Hopper-v4")
    >>> base_env.action_space
    Box(-1.0, 1.0, (3,), float32)
    >>> wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
    >>> wrapped_env.action_space
    Box(0.0, 1.0, (3,), float32)

You can access the environment underneath the **first** wrapper by using the :attr:`gymnasium.Wrapper.env` attribute.
As the :class:`gymnasium.Wrapper` class inherits from :class:`gymnasium.Env` then :attr:`gymnasium.Wrapper.env` can be another wrapper.

    >>> wrapped_env
    <RescaleAction<TimeLimit<OrderEnforcing<PassiveEnvChecker<HopperEnv<Hopper-v4>>>>>>
    >>> wrapped_env.env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<HopperEnv<Hopper-v4>>>>>

If you want to get to the environment underneath **all** of the layers of wrappers, you can use the
:attr:`gymnasium.Wrapper.unwrapped` attribute.
If the environment is already a bare environment, the :attr:`gymnasium.Wrapper.unwrapped` attribute will just return itself.

    >>> wrapped_env
    <RescaleAction<TimeLimit<OrderEnforcing<PassiveEnvChecker<HopperEnv<Hopper-v4>>>>>>
    >>> wrapped_env.unwrapped # doctest: +SKIP
    <gymnasium.envs.mujoco.hopper_v4.HopperEnv object at 0x7fbb5efd0490>

There are three common things you might want a wrapper to do:

- Transform actions before applying them to the base environment
- Transform observations that are returned by the base environment
- Transform rewards that are returned by the base environment

Such wrappers can be easily implemented by inheriting from :class:`gymnasium.ActionWrapper`,
:class:`gymnasium.ObservationWrapper`, or :class:`gymnasium.RewardWrapper` and implementing the respective transformation.
If you need a wrapper to do more complicated tasks, you can inherit from the :class:`gymnasium.Wrapper` class directly.

If you'd like to implement your own custom wrapper, check out `the corresponding tutorial <../../tutorials/gymnasium_basics/implementing_custom_wrappers>`_.
"""

# pyright: reportUnsupportedDunderAll=false
import importlib

from gymnasium.wrappers import vector
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.common import (
    Autoreset,
    OrderEnforcing,
    PassiveEnvChecker,
    RecordEpisodeStatistics,
    TimeLimit,
)
from gymnasium.wrappers.rendering import HumanRendering, RecordVideo, RenderCollection
from gymnasium.wrappers.stateful_action import StickyAction
from gymnasium.wrappers.stateful_observation import (
    DelayObservation,
    FrameStackObservation,
    MaxAndSkipObservation,
    NormalizeObservation,
    TimeAwareObservation,
)
from gymnasium.wrappers.stateful_reward import NormalizeReward
from gymnasium.wrappers.transform_action import (
    ClipAction,
    RescaleAction,
    TransformAction,
)
from gymnasium.wrappers.transform_observation import (
    AddRenderObservation,
    DtypeObservation,
    FilterObservation,
    FlattenObservation,
    GrayscaleObservation,
    RescaleObservation,
    ReshapeObservation,
    ResizeObservation,
    TransformObservation,
)
from gymnasium.wrappers.transform_reward import ClipReward, TransformReward


__all__ = [
    "vector",
    # --- Observation wrappers ---
    "AtariPreprocessing",
    "DelayObservation",
    "DtypeObservation",
    "FilterObservation",
    "FlattenObservation",
    "FrameStackObservation",
    "GrayscaleObservation",
    "TransformObservation",
    "MaxAndSkipObservation",
    "NormalizeObservation",
    "AddRenderObservation",
    "ResizeObservation",
    "ReshapeObservation",
    "RescaleObservation",
    "TimeAwareObservation",
    # --- Action Wrappers ---
    "ClipAction",
    "TransformAction",
    "RescaleAction",
    # "NanAction",
    "StickyAction",
    # --- Reward wrappers ---
    "ClipReward",
    "TransformReward",
    "NormalizeReward",
    # --- Common ---
    "TimeLimit",
    "Autoreset",
    "PassiveEnvChecker",
    "OrderEnforcing",
    "RecordEpisodeStatistics",
    # --- Rendering ---
    "RenderCollection",
    "RecordVideo",
    "HumanRendering",
    # --- Conversion ---
    "JaxToNumpy",
    "JaxToTorch",
    "NumpyToTorch",
]

# As these wrappers requires `jax` or `torch`, they are loaded by runtime for users trying to access them
#   to avoid `import jax` or `import torch` on `import gymnasium`.
_wrapper_to_class = {
    # data converters
    "JaxToNumpy": "jax_to_numpy",
    "JaxToTorch": "jax_to_torch",
    "NumpyToTorch": "numpy_to_torch",
}

_renamed_wrapper = {
    "AutoResetWrapper": "Autoreset",
    "FrameStack": "FrameStackObservation",
    "PixelObservationWrapper": "AddRenderObservation",
    "VectorListInfo": "vector.DictInfoToList",
}


def __getattr__(wrapper_name: str):
    """Load a wrapper by name.

    This optimizes the loading of gymnasium wrappers by only loading the wrapper if it is used.
    Errors will be raised if the wrapper does not exist or if the version is not the latest.

    Args:
        wrapper_name: The name of a wrapper to load.

    Returns:
        The specified wrapper.

    Raises:
        AttributeError: If the wrapper does not exist.
        DeprecatedWrapper: If the version is not the latest.
    """
    # Check if the requested wrapper is in the _wrapper_to_class dictionary
    if wrapper_name in _wrapper_to_class:
        import_stmt = f"gymnasium.wrappers.{_wrapper_to_class[wrapper_name]}"
        module = importlib.import_module(import_stmt)
        return getattr(module, wrapper_name)

    elif wrapper_name in _renamed_wrapper:
        raise AttributeError(
            f"{wrapper_name!r} has been renamed with `wrappers.{_renamed_wrapper[wrapper_name]}`"
        )

    raise AttributeError(f"module {__name__!r} has no attribute {wrapper_name!r}")
