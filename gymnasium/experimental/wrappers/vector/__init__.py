"""Wrappers for vector environments."""
# pyright: reportUnsupportedDunderAll=false
import importlib


__all__ = [
    # --- Vector only wrappers
    "VectoriseLambdaObservationV0",
    "VectoriseLambdaActionV0",
    "VectoriseLambdaRewardV0",
    "DictInfoToListV0",
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
    # "TimeAwareObservationV0",
    # "FrameStackObservationV0",
    # "DelayObservationV0",
    # --- Action Wrappers ---
    "LambdaActionV0",
    "ClipActionV0",
    "RescaleActionV0",
    # --- Reward wrappers ---
    "LambdaRewardV0",
    "ClipRewardV0",
    "NormalizeRewardV1",
    # --- Common ---
    "RecordEpisodeStatisticsV0",
    # --- Rendering ---
    # "RenderCollectionV0",
    # "RecordVideoV0",
    # "HumanRenderingV0",
    # --- Conversion ---
    "JaxToNumpyV0",
    "JaxToTorchV0",
    "NumpyToTorchV0",
]


_wrapper_to_class = {
    # --- dict_info_to_list
    "DictInfoToListV0": "dict_info_to_list",
    # --- vectorize_action.py
    "VectoriseLambdaActionV0": "vectorize_action",
    "LambdaActionV0": "vectorize_action",
    "ClipActionV0": "vectorize_action",
    "RescaleActionV0": "vectorize_action",
    # --- vectorize_observation.py
    "VectoriseLambdaObservationV0": "vectorize_observation",
    "LambdaObservationV0": "vectorize_observation",
    "FilterObservationV0": "vectorize_observation",
    "FlattenObservationV0": "vectorize_observation",
    "GrayscaleObservationV0": "vectorize_observation",
    "ResizeObservationV0": "vectorize_observation",
    "ReshapeObservationV0": "vectorize_observation",
    "RescaleObservationV0": "vectorize_observation",
    "DtypeObservationV0": "vectorize_observation",
    # --- vectorize_reward.py
    "VectoriseLambdaRewardV0": "vectorize_reward",
    "ClipRewardV0": "vectorize_reward",
    "LambdaRewardV0": "vectorize_reward",
    # --- stateful_action
    # --- stateful_observation
    # "TimeAwareObservationV0": "stateful_observation",
    # "DelayObservationV0": "stateful_observation",
    # "FrameStackObservationV0": "stateful_observation",
    # "NormalizeObservationV0": "stateful_observation",
    # "PixelObservationV0": "stateful_observation",
    # --- stateful_reward
    # "NormalizeRewardV1": "stateful_reward",
    # --- common
    "RecordEpisodeStatisticsV0": "record_episode_statistics",
    # --- rendering
    # "RenderCollectionV0": "rendering",
    # "RecordVideoV0": "rendering",
    # "HumanRenderingV0": "rendering",
    # --- jax_to_numpy
    "JaxToNumpyV0": "jax_to_numpy",
    # --- jax_to_torch
    "JaxToTorchV0": "jax_to_torch",
    # --- numpy_to_torch
    "NumpyToTorchV0": "numpy_to_torch",
}


def __getattr__(name: str):
    """To avoid having to load all vector wrappers on `import gymnasium` with all of their extra modules.

    This optimises the loading of gymnasium.

    Args:
        name: The name of a wrapper to load

    Returns:
        Wrapper
    """
    if name in _wrapper_to_class:
        import_stmt = (
            f"gymnasium.experimental.wrappers.vector.{_wrapper_to_class[name]}"
        )
        module = importlib.import_module(import_stmt)
        return getattr(module, name)
    # add helpful error message if version number has changed
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
