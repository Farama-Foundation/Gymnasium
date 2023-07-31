"""Module of wrapper classes.

Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly.
Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular. Wrappers can
also be chained to combine their effects.
Most environments that are generated via :meth:`gymnasium.make` will already be wrapped by default.

In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along
with (possibly optional) parameters to the wrapper's constructor.

    >>> import gymnasium as gym
    >>> from gymnasium.wrappers import RescaleActionV0
    >>> base_env = gym.make("Hopper-v4")
    >>> base_env.action_space
    Box(-1.0, 1.0, (3,), float32)
    >>> wrapped_env = RescaleActionV0(base_env, min_action=0, max_action=1)
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

If you'd like to implement your own custom wrapper, check out `the corresponding tutorial <../../tutorials/implementing_custom_wrappers>`_.
"""
# pyright: reportUnsupportedDunderAll=false
import importlib
import re

from gymnasium.error import DeprecatedWrapper
from gymnasium.wrappers import vector
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessingV0
from gymnasium.wrappers.common import (
    AutoresetV0,
    OrderEnforcingV0,
    PassiveEnvCheckerV0,
    RecordEpisodeStatisticsV0,
    TimeLimitV0,
)
from gymnasium.wrappers.lambda_action import (
    ClipActionV0,
    LambdaActionV0,
    RescaleActionV0,
)
from gymnasium.wrappers.lambda_observation import (
    DtypeObservationV0,
    FilterObservationV0,
    FlattenObservationV0,
    GrayscaleObservationV0,
    LambdaObservationV0,
    RenderObservationV0,
    RescaleObservationV0,
    ReshapeObservationV0,
    ResizeObservationV0,
)
from gymnasium.wrappers.lambda_reward import ClipRewardV0, LambdaRewardV0
from gymnasium.wrappers.rendering import (
    HumanRenderingV0,
    RecordVideoV0,
    RenderCollectionV0,
)
from gymnasium.wrappers.stateful_action import StickyActionV0
from gymnasium.wrappers.stateful_observation import (
    DelayObservationV0,
    FrameStackObservationV0,
    MaxAndSkipObservationV0,
    NormalizeObservationV0,
    TimeAwareObservationV0,
)
from gymnasium.wrappers.stateful_reward import NormalizeRewardV1


# Todo - Add legacy wrapper to new wrapper error for users when merged into gymnasium.wrappers


__all__ = [
    "vector",
    # --- Observation wrappers ---
    "AtariPreprocessingV0",
    "DelayObservationV0",
    "DtypeObservationV0",
    "FilterObservationV0",
    "FlattenObservationV0",
    "FrameStackObservationV0",
    "GrayscaleObservationV0",
    "LambdaObservationV0",
    "MaxAndSkipObservationV0",
    "NormalizeObservationV0",
    "RenderObservationV0",
    "ResizeObservationV0",
    "ReshapeObservationV0",
    "RescaleObservationV0",
    "TimeAwareObservationV0",
    # --- Action Wrappers ---
    "ClipActionV0",
    "LambdaActionV0",
    "RescaleActionV0",
    # "NanAction",
    "StickyActionV0",
    # --- Reward wrappers ---
    "ClipRewardV0",
    "LambdaRewardV0",
    "NormalizeRewardV1",
    # --- Common ---
    "TimeLimitV0",
    "AutoresetV0",
    "PassiveEnvCheckerV0",
    "OrderEnforcingV0",
    "RecordEpisodeStatisticsV0",
    # --- Rendering ---
    "RenderCollectionV0",
    "RecordVideoV0",
    "HumanRenderingV0",
    # --- Conversion ---
    "JaxToNumpyV0",
    "JaxToTorchV0",
    "NumpyToTorchV0",
]

# As these wrappers requires `jax` or `torch`, they are loaded by runtime for users trying to access them
#   to avoid `import jax` or `import torch` on `import gymnasium`.
_wrapper_to_class = {
    # data converters
    "JaxToNumpyV0": "jax_to_numpy",
    "JaxToTorchV0": "jax_to_torch",
    "NumpyToTorchV0": "numpy_to_torch",
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

    # Define a regex pattern to match the integer suffix (version number) of the wrapper
    int_suffix_pattern = r"(\d+)$"
    version_match = re.search(int_suffix_pattern, wrapper_name)

    # If a version number is found, extract it and the base wrapper name
    if version_match:
        version = int(version_match.group())
        base_name = wrapper_name[: -len(version_match.group())]
    else:
        version = float("inf")
        base_name = wrapper_name[:-2]

    # Filter the list of all wrappers to include only those with the same base name
    matching_wrappers = [name for name in __all__ if name.startswith(base_name)]

    # If no matching wrappers are found, raise an AttributeError
    if not matching_wrappers:
        raise AttributeError(f"module {__name__!r} has no attribute {wrapper_name!r}")

    # Find the latest version of the matching wrappers
    latest_wrapper = max(
        matching_wrappers, key=lambda s: int(re.findall(int_suffix_pattern, s)[0])
    )
    latest_version = int(re.findall(int_suffix_pattern, latest_wrapper)[0])

    # If the requested wrapper is an older version, raise a DeprecatedWrapper exception
    if version < latest_version:
        raise DeprecatedWrapper(
            f"{wrapper_name!r} is now deprecated, use {latest_wrapper!r} instead.\n"
            f"To see the changes made, go to "
            f"https://gymnasium.farama.org/api/wrappers/#gymnasium.wrappers.{latest_wrapper}"
        )
    # If the requested version is invalid, raise an AttributeError
    else:
        raise AttributeError(
            f"module {__name__!r} has no attribute {wrapper_name!r}, did you mean {latest_wrapper!r}"
        )
