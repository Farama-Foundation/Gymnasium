---
title: Experimental
---

# Experimental

```{toctree}
:hidden:
experimental/functional
experimental/wrappers
experimental/vector
experimental/vector_wrappers
```

## Functional Environments

The gymnasium ``Env`` provides high flexibility for the implementation of individual environments however this can complicate parallelism of environments. Therefore, we propose the :class:`gymnasium.experimental.FuncEnv` where each part of environment has its own function related to it. 

## Wrappers

Gymnasium already contains a large collection of wrappers, but we believe that the wrappers can be improved to

 * Support arbitrarily complex observation / action spaces. As RL has advanced, action and observation spaces are becoming more complex and the current wrappers were not implemented with these spaces in mind.
 * Support for numpy, jax and pytorch data. With hardware accelerated environments, i.e. Brax, written in Jax and similar pytorch based programs, numpy is not the only game in town anymore. Therefore, these upgrades will use Jumpy for calling numpy, jax and torch depending on the data.
 * More wrappers. Projects like Supersuit aimed to bring more wrappers for RL however wrappers can be moved into Gymnasium.
 * Versioning. Like environments, the implementation details of wrapper can cause changes agent performance. Therefore, we propose adding version numbers with all wrappers.

 * In v28, we aim to rewrite the VectorEnv to not inherit from Env, as a result new vectorised versions of the wrappers will be provided.

### Lambda Observation Wrappers
```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table:: 
    :header-rows: 1
    
    * - Old name
      - New name
      - Vector version
      - Tree structure
    * - :class:`wrappers.TransformObservation`
      - :class:`experimental.wrappers.LambdaObservationV0`
      - VectorLambdaObservation
      - No
    * - :class:`wrappers.FilterObservation`
      - FilterObservation
      - VectorFilterObservation (*)
      - Yes
    * - :class:`wrappers.FlattenObservation`
      - FlattenObservation
      - VectorFlattenObservation (*)
      - No
    * - :class:`wrappers.GrayScaleObservation`
      - GrayscaleObservation
      - VectorGrayscaleObservation (*)
      - Yes
    * - :class:`wrappers.PixelObservationWrapper`
      - PixelObservation 
      - VectorPixelObservation (*)
      - No
    * - :class:`wrappers.ResizeObservation`
      - ResizeObservation
      - VectorResizeObservation (*)
      - Yes
    * - Not Implemented
      - ReshapeObservation
      - VectorReshapeObservation (*)
      - Yes
    * - Not Implemented
      - RescaleObservation
      - VectorRescaleObservation (*)
      - Yes
    * - Not Implemented
      - DtypeObservation
      - VectorDtypeObservation (*)
      - Yes
    * - :class:`NormalizeObservation`
      - NormalizeObservation 
      - VectorNormalizeObservation
      - No
    * - :class:`TimeAwareObservation`
      - TimeAwareObservation
      - VectorTimeAwareObservation
      - No
    * - :class:`FrameStack`
      - FrameStackObservation
      - VectorFrameStackObservation
      - No
    * - Not Implemented
      - DelayObservation
      - VectorDelayObservation
      - No
    * - :class:`AtariPreprocessing`
      - AtariPreprocessing
      - Not Implemented
      - No
```

### Lambda Action Wrappers
```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table:: 
    :header-rows: 1
    
    * - Old name
      - New name
      - Vector version
      - Tree structure
    * - Not Implemented
      - :class:`experimental.wrappers.LambdaActionV0`
      - VectorLambdaAction
      - No
    * - :class:`wrappers.ClipAction`
      - ClipAction
      - VectorClipAction (*)
      - Yes
    * - :class:`wrappers.RescaleAction`
      - RescaleAction
      - VectorRescaleAction (*)
      - Yes
    * - Not Implemented
      - NanAction
      - VectorNanAction (*)
      - Yes
    * - Not Implemented
      - StickyAction
      - VectorStickyAction
      - No
```

### Lambda Reward Wrappers
```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table:: 
    :header-rows: 1
    
    * - Old name
      - New name
      - Vector version
    * - :class:`wrappers.TransformReward`
      - :class:`experimental.wrappers.LambdaRewardV0`
      - VectorLambdaReward
    * - Not Implemented
      - :class:`experimental.wrappers.ClipRewardV0`
      - VectorClipReward (*)
    * - Not Implemented
      - RescaleReward
      - VectorRescaleReward (*)
    * - :class:`wrappers.NormalizeReward`
      - NormalizeReward
      - VectorNormalizeReward
```

### Common Wrappers
```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table:: 
    :header-rows: 1
    
    * - Old name
      - New name
      - Vector version
    * - :class:`wrappers.AutoResetWrapper`
      - AutoReset
      - VectorAutoReset
    * - :class:`wrappers.PassiveEnvChecker`
      - PassiveEnvChecker
      - VectorPassiveEnvChecker
    * - :class:`wrappers.OrderEnforcing`
      - OrderEnforcing
      - VectorOrderEnforcing (*)  
    * - :class:`wrappers.EnvCompatibility`
      - Moved to `shimmy <https://github.com/Farama-Foundation/Shimmy/blob/main/shimmy/openai_gym_compatibility.py>`_
      - Not Implemented
    * - :class:`wrappers.RecordEpisodeStatistics`
      - RecordEpisodeStatistics
      - VectorRecordEpisodeStatistics
    * - :class:`wrappers.RenderCollection`
      - RenderCollection
      - VectorRenderCollection
    * - :class:`wrappers.HumanRendering`
      - HumanRendering
      - Not Implemented
    * - Not Implemented
      - :class:`experimental.wrappers.JaxToNumpy`
      - VectorJaxToNumpy (*)
    * - Not Implemented
      - :class:`experimental.wrappers.JaxToTorch`
      - VectorJaxToTorch (*)
```

### Vector Only Wrappers
```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table:: 
    :header-rows: 1
    
    * - Old name
      - New name
    * - :class:`wrappers.VectorListInfo`
      - VectorListInfo
```

## Vector Environment

These changes will be made in v0.28

## Wrappers for Vector Environments

These changes will be made in v0.28
