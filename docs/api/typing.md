---
title: Typing
---

# Typing

```{eval-rst}
.. automodule:: gymnasium.typing
```

For example, a custom environment producing image observations and accepting discrete actions, and a wrapper that converts those observations to grayscale, would be annotated as:

```python
import numpy as np

import gymnasium as gym
from gymnasium.typing import ActType, ObsType, WrapperObsType


class MyEnv(gym.Env[np.ndarray, int]):
    """An environment with `np.ndarray` observations and `int` actions."""


class GrayscaleWrapper(gym.ObservationWrapper[np.ndarray, ActType, np.ndarray]):
    """Transforms `(H, W, 3)` uint8 observations into `(H, W)` grayscale ones."""

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.mean(observation, axis=-1).astype(np.uint8)
```

Every TypeVar defaults to ``Any``, so ``gym.Env``, ``gym.Wrapper[np.ndarray, int]`` and other partial subscriptions remain valid.

## Single-environment vocabulary

```{eval-rst}
.. autodata:: gymnasium.typing.ObsType
   :no-value:
.. autodata:: gymnasium.typing.ActType
   :no-value:
.. autodata:: gymnasium.typing.RenderFrame
   :no-value:
.. autodata:: gymnasium.typing.WrapperObsType
   :no-value:
.. autodata:: gymnasium.typing.WrapperActType
   :no-value:
```

## Vector-environment vocabulary

```{eval-rst}
.. autodata:: gymnasium.typing.VectorObsType
   :no-value:
.. autodata:: gymnasium.typing.VectorActType
   :no-value:
.. autodata:: gymnasium.typing.RewardArrayType
   :no-value:
.. autodata:: gymnasium.typing.BoolArrayType
   :no-value:
```
