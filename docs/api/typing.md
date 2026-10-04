---
title: Typing
---

# Typing

```{warning}
These `TypeVar`s exist because Gymnasium supports Python versions without PEP 695 type parameter syntax. They will be replaced by PEP 695 syntax when support for those versions is dropped. Do not build long-lived abstractions on them.
```

{class}`gymnasium.Env` and {class}`gymnasium.vector.VectorEnv` are generic in the types of their observations and actions, and the wrappers are generic in both their own types and the wrapped environment's. For example, an environment with image observations and discrete actions, and a wrapper that converts those observations to grayscale, would be annotated as:

```python
import numpy as np

import gymnasium as gym
from gymnasium.core import ActType
from gymnasium.spaces import Box


class MyEnv(gym.Env[np.ndarray, int]):
    """An environment with `np.ndarray` observations and `int` actions."""


class GrayscaleWrapper(gym.ObservationWrapper[np.ndarray, ActType, np.ndarray]):
    """Transforms `(H, W, 3)` uint8 observations into `(H, W)` grayscale ones."""

    def __init__(self, env: gym.Env[np.ndarray, ActType]):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)
        self.observation_space = Box(
            0, 255, env.observation_space.shape[:-1], dtype=np.uint8
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.mean(observation, axis=-1).astype(np.uint8)
```

The `TypeVar`s have no defaults, so a class must be given either all of its type arguments or none. Without any, such as a bare `gym.Env` or `VectorWrapper`, every type parameter is `Any`. A partial subscription, such as `gym.Wrapper[np.ndarray, int]`, is an error.

## Single-environment types

```{eval-rst}
.. autodata:: gymnasium.core.ObsType
   :no-value:
.. autodata:: gymnasium.core.ActType
   :no-value:
.. autodata:: gymnasium.core.WrapperObsType
   :no-value:
.. autodata:: gymnasium.core.WrapperActType
   :no-value:
.. autodata:: gymnasium.core.RenderFrame
   :no-value:
```

## Vector-environment types

Vector environments and wrappers reuse the single-environment `TypeVar`s for their batched observations and actions, with one additional parameter for the rewards, terminations and truncations arrays returned by `step`.

```{eval-rst}
.. autodata:: gymnasium.vector.vector_env.ArrayType
   :no-value:
```

## Wrapper type parameters

Each vector wrapper has the same type parameters as its single-environment equivalent, followed by `ArrayType`. The `Wrapper` parameters are the types the wrapper exposes, and the others are the wrapped environment's:

| Single environment                                           | Vector environment                                                            |
|--------------------------------------------------------------|-------------------------------------------------------------------------------|
| `Env[ObsType, ActType]`                                      | `VectorEnv[ObsType, ActType, ArrayType]`                                      |
| `Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]`  | `VectorWrapper[WrapperObsType, WrapperActType, ObsType, ActType, ArrayType]`  |
| `ObservationWrapper[WrapperObsType, ActType, ObsType]`       | `VectorObservationWrapper[WrapperObsType, ActType, ObsType, ArrayType]`       |
| `ActionWrapper[ObsType, WrapperActType, ActType]`            | `VectorActionWrapper[ObsType, WrapperActType, ActType, ArrayType]`            |
| `RewardWrapper[ObsType, ActType]`                            | `VectorRewardWrapper[ObsType, ActType, ArrayType]`                            |

For example, the vector version of the grayscale wrapper above:

```python
import numpy as np

from gymnasium.core import ActType
from gymnasium.vector import VectorObservationWrapper
from gymnasium.vector.vector_env import ArrayType


class VectorGrayscaleWrapper(
    VectorObservationWrapper[np.ndarray, ActType, np.ndarray, ArrayType]
):
    """Transforms `(N, H, W, 3)` uint8 observations into `(N, H, W)` grayscale ones."""

    def observations(self, observations: np.ndarray) -> np.ndarray:
        return np.mean(observations, axis=-1).astype(np.uint8)
```

The base wrappers' `reset` and `step` pass the wrapped environment's data through unchanged, so a wrapper that changes the observation or action type must override them (or use {class}`~gymnasium.vector.VectorObservationWrapper` or {class}`~gymnasium.vector.VectorActionWrapper`, which do).

## Built-in vector wrappers

The vector wrappers that don't change the observations or actions are generic in the wrapped environment's types, so they keep them:

| Wrapper                                                                                                                                                                             | Type parameters                                                                 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `HumanRendering`, {class}`~gymnasium.wrappers.vector.DictInfoToList`                                                                                                                | `[ObsType, ActType, ArrayType]`                                                 |
| {class}`~gymnasium.wrappers.vector.RecordEpisodeStatistics`, {class}`~gymnasium.wrappers.vector.NormalizeReward`, `RecordVideo`                                                     | `[ObsType, ActType]`, with `ArrayType` as `np.ndarray`                          |
| {class}`~gymnasium.wrappers.vector.NormalizeObservation`                                                                                                                            | `[ActType, ArrayType]`, with observations as `np.ndarray`                       |
| `ArrayConversion` (the base of {class}`~gymnasium.wrappers.vector.JaxToNumpy`, {class}`~gymnasium.wrappers.vector.JaxToTorch` and {class}`~gymnasium.wrappers.vector.NumpyToTorch`) | `[WrapperObsType, WrapperActType, ObsType, ActType]`, with `ArrayType` as `Any` |

{class}`~gymnasium.wrappers.vector.TransformObservation`, {class}`~gymnasium.wrappers.vector.TransformAction` and {class}`~gymnasium.wrappers.vector.TransformReward` are generic in their function's input and output types, and the remaining vector wrappers, including the conversion wrappers above, aren't generic, so their wrapped environment's types are `Any`.
