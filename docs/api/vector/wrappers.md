---
title: Vector Wrappers
---

# Wrappers

## Autoreset mode support

Vector wrappers don't all support every [autoreset mode](https://farama.org/Vector-Autoreset-Mode), read from the wrapped environment's `metadata["autoreset_mode"]`. Wrappers raise a `ValueError` at construction for an unsupported mode.

| Wrapper | Next-step | Same-step | Disabled |
|---------|:---------:|:---------:|:--------:|
| `VectorizeTransformObservation` and its subclasses (`FilterObservation`, `FlattenObservation`, `GrayscaleObservation`, `ResizeObservation`, `ReshapeObservation`, `RescaleObservation`, `DtypeObservation`) | ✓ | ✓ (`info["final_obs"]` is transformed for each sub-environment) | ✓ |
| `TransformObservation` and other `VectorObservationWrapper` subclasses | ✓ | ✗ (`info["final_obs"]` is not transformed) | ✓ |
| `NormalizeObservation` | ✓ | ✓ (`info["final_obs"]` is normalized and included in the statistics) | ✓ (only resets of all sub-environments, partial `reset_mask` resets are rejected) |
| Action wrappers (`VectorActionWrapper` subclasses, including `VectorizeTransformAction`) | ✓ | ✓ | ✓ |
| Reward wrappers (`VectorRewardWrapper` subclasses, including `VectorizeTransformReward`) | ✓ | ✓ | ✓ |
| `NormalizeReward` | ✓ | ✓ | ✓ |
| `RecordEpisodeStatistics` | ✓ | ✓ | ✓ |
| `DictInfoToList` | ✓ | ✓ | ✓ |
| `RecordVideo` | ✓ | ✓ (recorded episodes don't include the final frame) | ✓ |

Unlike the single-environment `NormalizeReward`, the vector `NormalizeReward` restarts a sub-environment's accumulated return once it is reset, i.e., at the end of an episode for same-step autoreset and on `reset` for disabled autoreset.

For next-step autoreset, reward wrappers return the reward of a sub-environment's autoreset step unmodified, as no environment step occurs.

```{eval-rst}
.. autoclass:: gymnasium.vector.VectorWrapper

    .. automethod:: gymnasium.vector.VectorWrapper.step
    .. automethod:: gymnasium.vector.VectorWrapper.reset
    .. automethod:: gymnasium.vector.VectorWrapper.render
    .. automethod:: gymnasium.vector.VectorWrapper.close

.. autoclass:: gymnasium.vector.VectorObservationWrapper

    .. automethod:: gymnasium.vector.VectorObservationWrapper.observations

.. autoclass:: gymnasium.vector.VectorActionWrapper

    .. automethod:: gymnasium.vector.VectorActionWrapper.actions

.. autoclass:: gymnasium.vector.VectorRewardWrapper

    .. automethod:: gymnasium.vector.VectorRewardWrapper.rewards
```

## Vector Only wrappers

```{eval-rst}
.. autoclass:: gymnasium.wrappers.vector.DictInfoToList

.. autoclass:: gymnasium.wrappers.vector.VectorizeTransformObservation
.. autoclass:: gymnasium.wrappers.vector.VectorizeTransformAction
.. autoclass:: gymnasium.wrappers.vector.VectorizeTransformReward
```

## Vectorized Common wrappers

```{eval-rst}
.. autoclass:: gymnasium.wrappers.vector.RecordEpisodeStatistics
```

## Implemented Observation wrappers

```{eval-rst}
.. autoclass:: gymnasium.wrappers.vector.TransformObservation
.. autoclass:: gymnasium.wrappers.vector.FilterObservation
.. autoclass:: gymnasium.wrappers.vector.FlattenObservation
.. autoclass:: gymnasium.wrappers.vector.GrayscaleObservation
.. autoclass:: gymnasium.wrappers.vector.ResizeObservation
.. autoclass:: gymnasium.wrappers.vector.ReshapeObservation
.. autoclass:: gymnasium.wrappers.vector.RescaleObservation
.. autoclass:: gymnasium.wrappers.vector.DtypeObservation
.. autoclass:: gymnasium.wrappers.vector.NormalizeObservation
```

## Implemented Action wrappers

```{eval-rst}
.. autoclass:: gymnasium.wrappers.vector.TransformAction
.. autoclass:: gymnasium.wrappers.vector.ClipAction
.. autoclass:: gymnasium.wrappers.vector.RescaleAction
```

## Implemented Reward wrappers

```{eval-rst}
.. autoclass:: gymnasium.wrappers.vector.TransformReward
.. autoclass:: gymnasium.wrappers.vector.ClipReward
.. autoclass:: gymnasium.wrappers.vector.NormalizeReward
```

## Implemented Data Conversion wrappers

```{eval-rst}
.. autoclass:: gymnasium.wrappers.vector.ArrayConversion
.. autoclass:: gymnasium.wrappers.vector.JaxToNumpy
.. autoclass:: gymnasium.wrappers.vector.JaxToTorch
.. autoclass:: gymnasium.wrappers.vector.NumpyToTorch
```
