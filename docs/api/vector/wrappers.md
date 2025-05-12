---
title: Vector Wrappers
---

# Wrappers

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
