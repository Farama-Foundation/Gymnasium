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

```{eval-rst}
The gymnasium ``Env`` provides high flexibility for the implementation of individual environments however this can complicate parallelism of environments. Therefore, we propose the :class:`gymnasium.experimental.FuncEnv` where each part of environment has its own function related to it.
```

## Wrappers

Gymnasium already contains a large collection of wrappers, but we believe that the wrappers can be improved to

 * Support arbitrarily complex observation / action spaces. As RL has advanced, action and observation spaces are becoming more complex and the current wrappers were not implemented with these spaces in mind.
 * Support for numpy, jax and pytorch data. With hardware accelerated environments, i.e. Brax, written in Jax and similar pytorch based programs, numpy is not the only game in town anymore. Therefore, these upgrades will use Jumpy for calling numpy, jax and torch depending on the data.
 * More wrappers. Projects like Supersuit aimed to bring more wrappers for RL however wrappers can be moved into Gymnasium.
 * Versioning. Like environments, the implementation details of wrapper can cause changes agent performance. Therefore, we propose adding version numbers with all wrappers.

 * In v28, we aim to rewrite the VectorEnv to not inherit from Env, as a result new vectorised versions of the wrappers will be provided.

### Observation Wrappers
```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table::
    :header-rows: 1

    * - Old name
      - New name
    * - :class:`wrappers.TransformObservation`
      - :class:`experimental.wrappers.LambdaObservationV0`
    * - :class:`wrappers.FilterObservation`
      - :class:`experimental.wrappers.FilterObservationV0`
    * - :class:`wrappers.FlattenObservation`
      - :class:`experimental.wrappers.FlattenObservationV0`
    * - :class:`wrappers.GrayScaleObservation`
      - :class:`experimental.wrappers.GrayscaleObservationV0`
    * - :class:`wrappers.ResizeObservation`
      - :class:`experimental.wrappers.ResizeObservationV0`
    * - ``supersuit.reshape_v0``
      - :class:`experimental.wrappers.ReshapeObservationV0`
    * - Not Implemented
      - :class:`experimental.wrappers.RescaleObservationV0`
    * - ``supersuit.dtype_v0``
      - :class:`experimental.wrappers.DtypeObservationV0`
    * - :class:`wrappers.PixelObservationWrapper`
      - :class:`experimental.wrappers.PixelObservationV0`
    * - :class:`wrappers.NormalizeObservation`
      - :class:`experimental.wrappers.NormalizeObservationV0`
    * - :class:`wrappers.TimeAwareObservation`
      - :class:`experimental.wrappers.TimeAwareObservationV0`
    * - :class:`wrappers.FrameStack`
      - :class:`experimental.wrappers.FrameStackObservationV0`
    * - ``supersuit.delay_observations_v0``
      - :class:`experimental.wrappers.DelayObservationV0`
    * - :class:`wrappers.AtariPreprocessing`
      - :class:`experimental.wrappers.AtariPreprocessingV0`
```

### Action Wrappers
```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table::
    :header-rows: 1

    * - Old name
      - New name
    * - ``supersuit.action_lambda_v1``
      - :class:`experimental.wrappers.LambdaActionV0`
    * - :class:`wrappers.ClipAction`
      - :class:`experimental.wrappers.ClipActionV0`
    * - :class:`wrappers.RescaleAction`
      - :class:`experimental.wrappers.RescaleActionV0`
    * - ``supersuit.sticky_actions_v0``
      - :class:`experimental.wrappers.StickyActionV0`
```

### Reward Wrappers
```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table::
    :header-rows: 1

    * - Old name
      - New name
    * - :class:`wrappers.TransformReward`
      - :class:`experimental.wrappers.LambdaRewardV0`
    * - ``supersuit.clip_reward_v0``
      - :class:`experimental.wrappers.ClipRewardV0`
    * - :class:`wrappers.NormalizeReward`
      - :class:`experimental.wrappers.NormalizeRewardV0`
```

### Common Wrappers

```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table::
    :header-rows: 1

    * - Old name
      - New name
    * - :class:`wrappers.AutoResetWrapper`
      - :class:`experimental.wrappers.AutoresetV0`
    * - :class:`wrappers.PassiveEnvChecker`
      - :class:`experimental.wrappers.PassiveEnvCheckerV0`
    * - :class:`wrappers.OrderEnforcing`
      - :class:`experimental.wrappers.OrderEnforcingV0`
    * - :class:`wrappers.EnvCompatibility`
      - Moved to `shimmy <https://github.com/Farama-Foundation/Shimmy/blob/main/shimmy/openai_gym_compatibility.py>`_
    * - :class:`wrappers.RecordEpisodeStatistics`
      - :class:`experimental.wrappers.RecordEpisodeStatisticsV0`
```

### Rendering Wrappers

```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table::
    :header-rows: 1

    * - Old name
      - New name
    * - :class:`wrapper.RecordVideo`
      - :class:`experimental.wrappers.RecordVideoV0`
    * - :class:`wrappers.HumanRendering`
      - :class:`experimental.wrappers.HumanRenderingV0`
    * - :class:`wrappers.RenderCollection`
      - :class:`experimental.wrappers.RenderCollectionV0`
```

### Environment data conversion

```{eval-rst}
.. py:currentmodule:: gymnasium

* :class:`experimental.wrappers.JaxToNumpyV0`
* :class:`experimental.wrappers.JaxToTorchV0`
* :class:`experimental.wrappers.NumpyToTorchV0`
```

## Vector Environment

These changes will be made in v0.28

## Wrappers for Vector Environments

These changes will be made in v0.28
