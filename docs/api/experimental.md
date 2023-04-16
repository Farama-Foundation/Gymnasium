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
experimental/vector_utils
```

## Functional Environments

```{eval-rst}
The gymnasium ``Env`` provides high flexibility for the implementation of individual environments however this can complicate parallelism of environments. Therefore, we propose the :class:`gymnasium.experimental.FuncEnv` where each part of environment has its own function related to it.
```

## Wrappers

Gymnasium already contains a large collection of wrappers, but we believe that the wrappers can be improved to

* (Work in progress) Support arbitrarily complex observation / action spaces. As RL has advanced, action and observation spaces are becoming more complex and the current wrappers were not implemented with this mind.
* Support for Jax-based environments. With hardware accelerated environments, i.e. Brax, written in Jax and similar PyTorch based programs, NumPy is not the only game in town anymore. Therefore, these upgrades will use [Jumpy](https://github.com/farama-Foundation/jumpy), a project developed by Farama Foundation to provide automatic compatibility for NumPy, Jax and in the future PyTorch data for a large subset of the NumPy functions.
* More wrappers. Projects like [Supersuit](https://github.com/farama-Foundation/supersuit) aimed to bring more wrappers for RL, however, many users were not aware of the wrappers, so we plan to move the wrappers into Gymnasium. If we are missing common wrappers from the list provided above, please create an issue.
* Versioning. Like environments, the implementation details of wrappers can cause changes in agent performance. Therefore, we propose adding version numbers to all wrappers, i.e., `LambaActionV0`. We don't expect these version numbers to change regularly similar to environment version numbers and should ensure that all users know when significant changes could affect your agent's performance. Additionally, we hope that this will improve reproducibility of RL in the future, this is critical for academia.
* In v28, we aim to rewrite the VectorEnv to not inherit from Env, as a result new vectorized versions of the wrappers will be provided.

We aimed to replace the wrappers in gymnasium v0.30.0 with these experimental wrappers.

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
    * - `supersuit.reshape_v0 <https://github.com/Farama-Foundation/SuperSuit/blob/314831a7d18e7254f455b181694c049908f95155/supersuit/generic_wrappers/basic_wrappers.py#L40>`_
      - :class:`experimental.wrappers.ReshapeObservationV0`
    * - Not Implemented
      - :class:`experimental.wrappers.RescaleObservationV0`
    * - `supersuit.dtype_v0 <https://github.com/Farama-Foundation/SuperSuit/blob/314831a7d18e7254f455b181694c049908f95155/supersuit/generic_wrappers/basic_wrappers.py#L32>`_
      - :class:`experimental.wrappers.DtypeObservationV0`
    * - :class:`wrappers.PixelObservationWrapper`
      - :class:`experimental.wrappers.PixelObservationV0`
    * - :class:`wrappers.NormalizeObservation`
      - :class:`experimental.wrappers.NormalizeObservationV0`
    * - :class:`wrappers.TimeAwareObservation`
      - :class:`experimental.wrappers.TimeAwareObservationV0`
    * - :class:`wrappers.FrameStack`
      - :class:`experimental.wrappers.FrameStackObservationV0`
    * - `supersuit.delay_observations_v0 <https://github.com/Farama-Foundation/SuperSuit/blob/314831a7d18e7254f455b181694c049908f95155/supersuit/generic_wrappers/delay_observations.py#L6>`_
      - :class:`experimental.wrappers.DelayObservationV0`
```

### Action Wrappers
```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table::
    :header-rows: 1

    * - Old name
      - New name
    * - `supersuit.action_lambda_v1 <https://github.com/Farama-Foundation/SuperSuit/blob/314831a7d18e7254f455b181694c049908f95155/supersuit/lambda_wrappers/action_lambda.py#L73>`_
      - :class:`experimental.wrappers.LambdaActionV0`
    * - :class:`wrappers.ClipAction`
      - :class:`experimental.wrappers.ClipActionV0`
    * - :class:`wrappers.RescaleAction`
      - :class:`experimental.wrappers.RescaleActionV0`
    * - `supersuit.sticky_actions_v0 <https://github.com/Farama-Foundation/SuperSuit/blob/314831a7d18e7254f455b181694c049908f95155/supersuit/generic_wrappers/sticky_actions.py#L6>`_
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
    * - `supersuit.clip_reward_v0 <https://github.com/Farama-Foundation/SuperSuit/blob/314831a7d18e7254f455b181694c049908f95155/supersuit/generic_wrappers/basic_wrappers.py#L74>`_
      - :class:`experimental.wrappers.ClipRewardV0`
    * - :class:`wrappers.NormalizeReward`
      - :class:`experimental.wrappers.NormalizeRewardV1`
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
    * - :class:`wrappers.AtariPreprocessing`
      - :class:`experimental.wrappers.AtariPreprocessingV0`
```

### Rendering Wrappers

```{eval-rst}
.. py:currentmodule:: gymnasium

.. list-table::
    :header-rows: 1

    * - Old name
      - New name
    * - :class:`wrappers.RecordVideo`
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
