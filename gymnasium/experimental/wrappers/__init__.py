"""`__init__` for experimental wrappers, to avoid loading the wrappers if unnecessary, we can hack python."""
# pyright: reportUnsupportedDunderAll=false

import importlib


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
    "NormalizeRewardV0",
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
    "NormalizeRewardV0": "lambda_reward",
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


def __getattr__(name: str):
    """To avoid having to load all wrappers on `import gymnasium` with all of their extra modules.

    This optimises the loading of gymnasium.

    Args:
        name: The name of a wrapper to load

    Returns:
        Wrapper
    """
    if name in _wrapper_to_class:
        import_stmt = f"gymnasium.experimental.wrappers.{_wrapper_to_class[name]}"
        module = importlib.import_module(import_stmt)
        return getattr(module, name)
    # add helpful error message if version number has changed
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
