# Typing

#### WARNING
These `TypeVar`s exist because Gymnasium supports Python versions without PEP 695 type parameter syntax. They will be replaced by PEP 695 syntax when support for those versions is dropped. Do not build long-lived abstractions on them.

[`gymnasium.Env`](env.md#gymnasium.Env) and [`gymnasium.vector.VectorEnv`](vector.md#gymnasium.vector.VectorEnv) are generic in the types of their observations and actions, and the wrappers are generic in both their own types and the wrapped environment’s. For example, an environment with image observations and discrete actions, and a wrapper that converts those observations to grayscale, would be annotated as:

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

Every `TypeVar` defaults to `Any` ([PEP 696](https://peps.python.org/pep-0696/)), so a class may be given as few or as many of its type arguments as desired and the omitted ones fall back to `Any`. A bare `gym.Env` or `VectorWrapper` has every type parameter as `Any`, and a partial subscription such as `gym.Env[np.ndarray]` has `np.ndarray` observations and `Any` actions. On Python versions before 3.13, the defaults are provided by `typing-extensions >= 4.12`.

## Single-environment types

### gymnasium.core.ObsType

Type variable.

The preferred way to construct a type variable is via the dedicated
syntax for generic functions, classes, and type aliases:

```default
class Sequence[T]:  # T is a TypeVar
    ...
```

This syntax can also be used to create bound and constrained type
variables:

```default
# S is a TypeVar bound to str
class StrSequence[S: str]:
    ...

# A is a TypeVar constrained to str or bytes
class StrOrBytesSequence[A: (str, bytes)]:
    ...
```

However, if desired, reusable type variables can also be constructed
manually, like so:

```default
T = TypeVar('T')  # Can be anything
S = TypeVar('S', bound=str)  # Can be any subtype of str
A = TypeVar('A', str, bytes)  # Must be exactly str or bytes
```

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function and type alias definitions.

The variance of type variables is inferred by type checkers when they
are created through the type parameter syntax and when
`infer_variance=True` is passed. Manually created type variables may
be explicitly marked covariant or contravariant by passing
`covariant=True` or `contravariant=True`. By default, manually
created type variables are invariant. See PEP 484 and PEP 695 for more
details.

### gymnasium.core.ActType

Type variable.

The preferred way to construct a type variable is via the dedicated
syntax for generic functions, classes, and type aliases:

```default
class Sequence[T]:  # T is a TypeVar
    ...
```

This syntax can also be used to create bound and constrained type
variables:

```default
# S is a TypeVar bound to str
class StrSequence[S: str]:
    ...

# A is a TypeVar constrained to str or bytes
class StrOrBytesSequence[A: (str, bytes)]:
    ...
```

However, if desired, reusable type variables can also be constructed
manually, like so:

```default
T = TypeVar('T')  # Can be anything
S = TypeVar('S', bound=str)  # Can be any subtype of str
A = TypeVar('A', str, bytes)  # Must be exactly str or bytes
```

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function and type alias definitions.

The variance of type variables is inferred by type checkers when they
are created through the type parameter syntax and when
`infer_variance=True` is passed. Manually created type variables may
be explicitly marked covariant or contravariant by passing
`covariant=True` or `contravariant=True`. By default, manually
created type variables are invariant. See PEP 484 and PEP 695 for more
details.

### gymnasium.core.WrapperObsType

Type variable.

The preferred way to construct a type variable is via the dedicated
syntax for generic functions, classes, and type aliases:

```default
class Sequence[T]:  # T is a TypeVar
    ...
```

This syntax can also be used to create bound and constrained type
variables:

```default
# S is a TypeVar bound to str
class StrSequence[S: str]:
    ...

# A is a TypeVar constrained to str or bytes
class StrOrBytesSequence[A: (str, bytes)]:
    ...
```

However, if desired, reusable type variables can also be constructed
manually, like so:

```default
T = TypeVar('T')  # Can be anything
S = TypeVar('S', bound=str)  # Can be any subtype of str
A = TypeVar('A', str, bytes)  # Must be exactly str or bytes
```

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function and type alias definitions.

The variance of type variables is inferred by type checkers when they
are created through the type parameter syntax and when
`infer_variance=True` is passed. Manually created type variables may
be explicitly marked covariant or contravariant by passing
`covariant=True` or `contravariant=True`. By default, manually
created type variables are invariant. See PEP 484 and PEP 695 for more
details.

### gymnasium.core.WrapperActType

Type variable.

The preferred way to construct a type variable is via the dedicated
syntax for generic functions, classes, and type aliases:

```default
class Sequence[T]:  # T is a TypeVar
    ...
```

This syntax can also be used to create bound and constrained type
variables:

```default
# S is a TypeVar bound to str
class StrSequence[S: str]:
    ...

# A is a TypeVar constrained to str or bytes
class StrOrBytesSequence[A: (str, bytes)]:
    ...
```

However, if desired, reusable type variables can also be constructed
manually, like so:

```default
T = TypeVar('T')  # Can be anything
S = TypeVar('S', bound=str)  # Can be any subtype of str
A = TypeVar('A', str, bytes)  # Must be exactly str or bytes
```

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function and type alias definitions.

The variance of type variables is inferred by type checkers when they
are created through the type parameter syntax and when
`infer_variance=True` is passed. Manually created type variables may
be explicitly marked covariant or contravariant by passing
`covariant=True` or `contravariant=True`. By default, manually
created type variables are invariant. See PEP 484 and PEP 695 for more
details.

### gymnasium.core.RenderFrame *: TypeAlias*

Represent a PEP 604 union type

E.g. for int | str

## Vector-environment types

Vector environments and wrappers reuse the single-environment `TypeVar`s for their batched observations and actions, with one additional parameter for the rewards, terminations and truncations arrays returned by `step`.

### gymnasium.vector.vector_env.ArrayType

Type variable.

The preferred way to construct a type variable is via the dedicated
syntax for generic functions, classes, and type aliases:

```default
class Sequence[T]:  # T is a TypeVar
    ...
```

This syntax can also be used to create bound and constrained type
variables:

```default
# S is a TypeVar bound to str
class StrSequence[S: str]:
    ...

# A is a TypeVar constrained to str or bytes
class StrOrBytesSequence[A: (str, bytes)]:
    ...
```

However, if desired, reusable type variables can also be constructed
manually, like so:

```default
T = TypeVar('T')  # Can be anything
S = TypeVar('S', bound=str)  # Can be any subtype of str
A = TypeVar('A', str, bytes)  # Must be exactly str or bytes
```

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function and type alias definitions.

The variance of type variables is inferred by type checkers when they
are created through the type parameter syntax and when
`infer_variance=True` is passed. Manually created type variables may
be explicitly marked covariant or contravariant by passing
`covariant=True` or `contravariant=True`. By default, manually
created type variables are invariant. See PEP 484 and PEP 695 for more
details.

## Wrapper type parameters

Each vector wrapper has the same type parameters as its single-environment equivalent, followed by `ArrayType`. The `Wrapper` parameters are the types the wrapper exposes, and the others are the wrapped environment’s:

| Single environment                                          | Vector environment                                                           |
|-------------------------------------------------------------|------------------------------------------------------------------------------|
| `Env[ObsType, ActType]`                                     | `VectorEnv[ObsType, ActType, ArrayType]`                                     |
| `Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]` | `VectorWrapper[WrapperObsType, WrapperActType, ObsType, ActType, ArrayType]` |
| `ObservationWrapper[WrapperObsType, ActType, ObsType]`      | `VectorObservationWrapper[WrapperObsType, ActType, ObsType, ArrayType]`      |
| `ActionWrapper[ObsType, WrapperActType, ActType]`           | `VectorActionWrapper[ObsType, WrapperActType, ActType, ArrayType]`           |
| `RewardWrapper[ObsType, ActType]`                           | `VectorRewardWrapper[ObsType, ActType, ArrayType]`                           |

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

The base wrappers’ `reset` and `step` pass the wrapped environment’s data through unchanged, so a wrapper that changes the observation or action type must override them (or use [`VectorObservationWrapper`](vector/wrappers.md#gymnasium.vector.VectorObservationWrapper) or [`VectorActionWrapper`](vector/wrappers.md#gymnasium.vector.VectorActionWrapper), which do).

## Built-in vector environments

[`SyncVectorEnv`](vector/sync_vector_env.md#gymnasium.vector.SyncVectorEnv) and [`AsyncVectorEnv`](vector/async_vector_env.md#gymnasium.vector.AsyncVectorEnv) are generic in their observation and action types, `[ObsType, ActType]`, and return `np.ndarray` rewards, terminations and truncations, so they compose with the `np.ndarray` wrappers below. The `step` return types are more precise, `float64` rewards and `bool` terminations and truncations, than the `ArrayType` of `np.ndarray`, which is the common type of all three arrays.

## Built-in vector wrappers

The vector wrappers that transform the observations, actions or rewards mirror their single-environment equivalents, so they have the same type parameters, followed by `ArrayType`:

| Wrapper                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Type parameters                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`TransformObservation`](vector/wrappers.md#gymnasium.wrappers.vector.TransformObservation), [`VectorizeTransformObservation`](vector/wrappers.md#gymnasium.wrappers.vector.VectorizeTransformObservation), [`FilterObservation`](vector/wrappers.md#gymnasium.wrappers.vector.FilterObservation), [`FlattenObservation`](vector/wrappers.md#gymnasium.wrappers.vector.FlattenObservation), [`GrayscaleObservation`](vector/wrappers.md#gymnasium.wrappers.vector.GrayscaleObservation), [`ResizeObservation`](vector/wrappers.md#gymnasium.wrappers.vector.ResizeObservation), [`ReshapeObservation`](vector/wrappers.md#gymnasium.wrappers.vector.ReshapeObservation), [`RescaleObservation`](vector/wrappers.md#gymnasium.wrappers.vector.RescaleObservation), [`DtypeObservation`](vector/wrappers.md#gymnasium.wrappers.vector.DtypeObservation) | `[WrapperObsType, ActType, ObsType, ArrayType]`                   |
| [`TransformAction`](vector/wrappers.md#gymnasium.wrappers.vector.TransformAction), [`VectorizeTransformAction`](vector/wrappers.md#gymnasium.wrappers.vector.VectorizeTransformAction), [`ClipAction`](vector/wrappers.md#gymnasium.wrappers.vector.ClipAction), [`RescaleAction`](vector/wrappers.md#gymnasium.wrappers.vector.RescaleAction)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `[ObsType, WrapperActType, ActType, ArrayType]`                   |
| [`TransformReward`](vector/wrappers.md#gymnasium.wrappers.vector.TransformReward)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `[ObsType, ActType, ArrayType]`                                   |
| [`VectorizeTransformReward`](vector/wrappers.md#gymnasium.wrappers.vector.VectorizeTransformReward), [`ClipReward`](vector/wrappers.md#gymnasium.wrappers.vector.ClipReward)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `[ObsType, ActType, ArrayType]`, with `ArrayType` an `np.ndarray` |

`TransformObservation`, `TransformAction` and `TransformReward` infer the wrapper’s types from the `func` passed to them, e.g., `TransformObservation(envs, func)` with `envs: VectorEnv[Inner, ActType, ArrayType]` and `func: Callable[[Inner], Outer]` has `Outer` observations.

The vector wrappers that don’t change the observations or actions are generic in the wrapped environment’s types, so they keep them:

| Wrapper                                                                                                                                                                                                                                                                                                                          | Type parameters                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`HumanRendering`](vector/wrappers.md#gymnasium.wrappers.vector.HumanRendering), [`DictInfoToList`](vector/wrappers.md#gymnasium.wrappers.vector.DictInfoToList)                                                                                                                                                                 | `[ObsType, ActType, ArrayType]`                                                 |
| [`RecordEpisodeStatistics`](vector/wrappers.md#gymnasium.wrappers.vector.RecordEpisodeStatistics), [`NormalizeReward`](vector/wrappers.md#gymnasium.wrappers.vector.NormalizeReward), [`RecordVideo`](vector/wrappers.md#gymnasium.wrappers.vector.RecordVideo)                                                                  | `[ObsType, ActType]`, with `ArrayType` as `np.ndarray`                          |
| [`NormalizeObservation`](vector/wrappers.md#gymnasium.wrappers.vector.NormalizeObservation)                                                                                                                                                                                                                                      | `[ActType, ArrayType]`, with observations as `np.ndarray`                       |
| [`ArrayConversion`](vector/wrappers.md#gymnasium.wrappers.vector.ArrayConversion) (the base of [`JaxToNumpy`](vector/wrappers.md#gymnasium.wrappers.vector.JaxToNumpy), [`JaxToTorch`](vector/wrappers.md#gymnasium.wrappers.vector.JaxToTorch) and [`NumpyToTorch`](vector/wrappers.md#gymnasium.wrappers.vector.NumpyToTorch)) | `[WrapperObsType, WrapperActType, ObsType, ActType]`, with `ArrayType` as `Any` |

The remaining vector wrappers, including the conversion wrappers above, aren’t generic, so their wrapped environment’s types are `Any`.
