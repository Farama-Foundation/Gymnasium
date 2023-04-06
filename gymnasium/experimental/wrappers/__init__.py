"""`__init__` for experimental wrappers, to avoid loading the wrappers if unnecessary, we can hack python."""
# pyright: reportUnsupportedDunderAll=false
import importlib
import re
from typing import Any

from gymnasium.error import DeprecatedWrapper


__all__ = [
    "vector",
    # --- Observation wrappers ---
    "LambdaObservationV0",
    "FilterObservationV0",
    "FlattenObservationV0",
    "GrayscaleObservationV0",
    "ResizeObservationV0",
    "ReshapeObservationV0",
    "RescaleObservationV0",
    "DtypeObservationV0",
    "PixelObservationV0",
    "NormalizeObservationV0",
    "TimeAwareObservationV0",
    "FrameStackObservationV0",
    "DelayObservationV0",
    "AtariPreprocessingV0",
    # --- Action Wrappers ---
    "LambdaActionV0",
    "ClipActionV0",
    "RescaleActionV0",
    # "NanAction",
    "StickyActionV0",
    # --- Reward wrappers ---
    "LambdaRewardV0",
    "ClipRewardV0",
    "NormalizeRewardV1",
    # --- Common ---
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


_wrapper_to_class = {
    # lambda_action.py
    "LambdaActionV0": "lambda_action",
    "ClipActionV0": "lambda_action",
    "RescaleActionV0": "lambda_action",
    # lambda_observations.py
    "LambdaObservationV0": "lambda_observations",
    "FilterObservationV0": "lambda_observations",
    "FlattenObservationV0": "lambda_observations",
    "GrayscaleObservationV0": "lambda_observations",
    "ResizeObservationV0": "lambda_observations",
    "ReshapeObservationV0": "lambda_observations",
    "RescaleObservationV0": "lambda_observations",
    "DtypeObservationV0": "lambda_observations",
    "PixelObservationV0": "lambda_observations",
    "NormalizeObservationV0": "lambda_observations",
    # lambda_reward.py
    "ClipRewardV0": "lambda_reward",
    "LambdaRewardV0": "lambda_reward",
    "NormalizeRewardV1": "lambda_reward",
    # stateful_action
    "StickyActionV0": "stateful_action",
    # stateful_observation
    "TimeAwareObservationV0": "stateful_observation",
    "DelayObservationV0": "stateful_observation",
    "FrameStackObservationV0": "stateful_observation",
    # atari_preprocessing
    "AtariPreprocessingV0": "atari_preprocessing",
    # common
    "PassiveEnvCheckerV0": "common",
    "OrderEnforcingV0": "common",
    "AutoresetV0": "common",
    "RecordEpisodeStatisticsV0": "common",
    # rendering
    "RenderCollectionV0": "rendering",
    "RecordVideoV0": "rendering",
    "HumanRenderingV0": "rendering",
    # jax_to_numpy
    "JaxToNumpyV0": "jax_to_numpy",
    # "jax_to_numpy": "jax_to_numpy",
    # "numpy_to_jax": "jax_to_numpy",
    # jax_to_torch
    "JaxToTorchV0": "jax_to_torch",
    # "jax_to_torch": "jax_to_torch",
    # "torch_to_jax": "jax_to_torch",
    # numpy_to_torch
    "NumpyToTorchV0": "numpy_to_torch",
    # "torch_to_numpy": "numpy_to_torch",
    # "numpy_to_torch": "numpy_to_torch",
}


def __getattr__(wrapper_name: str) -> Any:
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
    try:
        version_str = re.findall(r"\d+", wrapper_name)[-1]
        num_digits = len(version_str)
        version = int(version_str)
    except IndexError:
        version = -1
        num_digits = 2

    base_name = wrapper_name[:-num_digits]

    # Get all wrappers that start with the base wrapper name
    wrappers = [name for name in __all__ if name.startswith(base_name)]

    # If the wrapper does not exist, raise an AttributeError
    if not wrappers:
        raise AttributeError(f"module {__name__!r} has no attribute {wrapper_name!r}")

    # Get the latest version of the wrapper
    latest_wrapper_name = sorted(
        wrappers, key=lambda s: int(re.findall(r"\d+", s)[-1])
    )[-1]
    latest_version = int(re.findall(r"\d+", latest_wrapper_name)[-1])

    # If the wrapper is the latest version, import it
    if wrapper_name is latest_wrapper_name:
        import_stmt = (
            f"gymnasium.experimental.wrappers.{_wrapper_to_class[wrapper_name]}"
        )
        module = importlib.import_module(import_stmt)
        return getattr(module, wrapper_name)

    # Raise an AttributeError exception if the version is wrong
    if version < 0 or version > latest_version:
        raise AttributeError(
            f"module {__name__!r} has no attribute {wrapper_name!r}, did you mean {latest_wrapper_name!r}"
        )

    # Raise a DeprecatedWrapper exception if the version is not the latest
    if version < latest_version:
        raise DeprecatedWrapper(
            f"{wrapper_name!r} is now deprecated, use {latest_wrapper_name!r} instead.\n"
            f"To see the changes made, go to "
            f"https://gymnasium.farama.org/api/experimental/wrappers/#gymnasium.experimental.wrappers.{latest_wrapper_name}"
        )
