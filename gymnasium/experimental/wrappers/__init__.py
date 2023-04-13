"""`__init__` for experimental wrappers, to avoid loading the wrappers if unnecessary, we can hack python."""
# pyright: reportUnsupportedDunderAll=false
import importlib
import re

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
        import_stmt = (
            f"gymnasium.experimental.wrappers.{_wrapper_to_class[wrapper_name]}"
        )
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
            f"https://gymnasium.farama.org/api/experimental/wrappers/#gymnasium.experimental.wrappers.{latest_wrapper}"
        )
    # If the requested version is invalid, raise an AttributeError
    else:
        raise AttributeError(
            f"module {__name__!r} has no attribute {wrapper_name!r}, did you mean {latest_wrapper!r}"
        )
