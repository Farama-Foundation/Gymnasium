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
