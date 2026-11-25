# Utility functions

## Vectorizing Spaces

### gymnasium.vector.utils.batch_space(space: [Space](../spaces.md#gymnasium.spaces.Space)[Any], n: int = 1) → [Space](../spaces.md#gymnasium.spaces.Space)[Any]

### gymnasium.vector.utils.batch_space(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box), n: int = 1) → [Box](../spaces/fundamental.md#gymnasium.spaces.Box)

### gymnasium.vector.utils.batch_space(space: [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete), n: int = 1) → [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete)

### gymnasium.vector.utils.batch_space(space: [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete), n: int = 1) → [Box](../spaces/fundamental.md#gymnasium.spaces.Box)

### gymnasium.vector.utils.batch_space(space: [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), n: int = 1) → [Box](../spaces/fundamental.md#gymnasium.spaces.Box)

### gymnasium.vector.utils.batch_space(space: [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple), n: int = 1) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

### gymnasium.vector.utils.batch_space(space: [Dict](../spaces/composite.md#gymnasium.spaces.Dict), n: int = 1) → [Dict](../spaces/composite.md#gymnasium.spaces.Dict)

### gymnasium.vector.utils.batch_space(space: [Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Text](../spaces/fundamental.md#gymnasium.spaces.Text) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence) | [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf), n: int = 1) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

### gymnasium.vector.utils.batch_space(space: [Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Text](../spaces/fundamental.md#gymnasium.spaces.Text) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence) | [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf), n: int = 1) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

### gymnasium.vector.utils.batch_space(space: [Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Text](../spaces/fundamental.md#gymnasium.spaces.Text) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence) | [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf), n: int = 1) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

### gymnasium.vector.utils.batch_space(space: [Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Text](../spaces/fundamental.md#gymnasium.spaces.Text) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence) | [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf), n: int = 1) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

### gymnasium.vector.utils.batch_space(space: [Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Text](../spaces/fundamental.md#gymnasium.spaces.Text) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence) | [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf), n: int = 1) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

Batch spaces of size n optimized for neural networks.

* **Parameters:**
  * **space** – Space (e.g. the observation space for a single environment in the vectorized environment).
  * **n** – Number of spaces to batch by (e.g. the number of environments in a vectorized environment).
* **Returns:**
  **Batched space of size \`n\`.**
* **Raises:**
  **ValueError** – Cannot batch spaces that does not have a registered function.

### Example

```pycon
>>> from gymnasium.spaces import Box, Dict
>>> import numpy as np
>>> space = Dict({
...     'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
...     'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)
... })
>>> batch_space(space, n=5)
Dict('position': Box(0.0, 1.0, (5, 3), float32), 'velocity': Box(0.0, 1.0, (5, 2), float32))
```

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[Space](../spaces.md#gymnasium.spaces.Space)]) → [Space](../spaces.md#gymnasium.spaces.Space)

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[Box](../spaces/fundamental.md#gymnasium.spaces.Box)]) → [Box](../spaces/fundamental.md#gymnasium.spaces.Box)

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete)]) → [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete)

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete)]) → [Box](../spaces/fundamental.md#gymnasium.spaces.Box)

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary)]) → [Box](../spaces/fundamental.md#gymnasium.spaces.Box)

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)]) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[Dict](../spaces/composite.md#gymnasium.spaces.Dict)]) → [Dict](../spaces/composite.md#gymnasium.spaces.Dict)

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Text](../spaces/fundamental.md#gymnasium.spaces.Text) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence) | [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf)]) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Text](../spaces/fundamental.md#gymnasium.spaces.Text) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence) | [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf)]) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Text](../spaces/fundamental.md#gymnasium.spaces.Text) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence) | [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf)]) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

### gymnasium.vector.utils.batch_differing_spaces(spaces: Sequence[[Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Text](../spaces/fundamental.md#gymnasium.spaces.Text) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence) | [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf)]) → [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple)

Batch a Sequence of spaces where subspaces to contain minor differences.

* **Parameters:**
  **spaces** – A sequence of Spaces with minor differences (the same space type but different parameters).
* **Returns:**
  **A batched space**

For Discrete spaces, dtypes are promoted to a common integer dtype that can
represent the full range of each input dtype. Combinations without such a
dtype, including uint64 mixed with a signed integer dtype, raise ValueError.

### Example

```pycon
>>> from gymnasium.spaces import Discrete
>>> spaces = [Discrete(3), Discrete(5), Discrete(4), Discrete(8)]
>>> batch_differing_spaces(spaces)
MultiDiscrete([3 5 4 8])
```

### gymnasium.vector.utils.concatenate(space: [Space](../spaces.md#gymnasium.spaces.Space), items: Iterable, out: tuple[Any, ...] | Mapping[str, Any] | ndarray) → tuple[Any, ...] | dict[str, Any] | ndarray

### gymnasium.vector.utils.concatenate(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), items: Iterable, out: ndarray) → ndarray

### gymnasium.vector.utils.concatenate(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), items: Iterable, out: ndarray) → ndarray

### gymnasium.vector.utils.concatenate(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), items: Iterable, out: ndarray) → ndarray

### gymnasium.vector.utils.concatenate(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), items: Iterable, out: ndarray) → ndarray

### gymnasium.vector.utils.concatenate(space: [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple), items: Iterable, out: tuple[Any, ...]) → tuple[Any, ...]

### gymnasium.vector.utils.concatenate(space: [Dict](../spaces/composite.md#gymnasium.spaces.Dict), items: Iterable, out: Mapping[str, Any]) → dict[str, Any]

### gymnasium.vector.utils.concatenate(space: [Space](../spaces.md#gymnasium.spaces.Space), items: Iterable[\_T], out: None) → tuple[\_T, ...]

### gymnasium.vector.utils.concatenate(space: [Space](../spaces.md#gymnasium.spaces.Space), items: Iterable[\_T], out: None) → tuple[\_T, ...]

### gymnasium.vector.utils.concatenate(space: [Space](../spaces.md#gymnasium.spaces.Space), items: Iterable[\_T], out: None) → tuple[\_T, ...]

### gymnasium.vector.utils.concatenate(space: [Space](../spaces.md#gymnasium.spaces.Space), items: Iterable[\_T], out: None) → tuple[\_T, ...]

### gymnasium.vector.utils.concatenate(space: [Space](../spaces.md#gymnasium.spaces.Space), items: Iterable[\_T], out: None) → tuple[\_T, ...]

Concatenate multiple samples from space into a single object.

* **Parameters:**
  * **space** – Space of each item (e.g. single_action_space from vectorized environment)
  * **items** – Samples to be concatenated (e.g. all sample should be an element of the space).
  * **out** – The output object (e.g. generated from create_empty_array)
* **Returns:**
  **The output object, can be the same object \`out\`.**
* **Raises:**
  **ValueError** – Space is not a valid `gymnasium.Space` instance

### Example

```pycon
>>> from gymnasium.spaces import Box
>>> import numpy as np
>>> space = Box(low=0, high=1, shape=(3,), seed=42, dtype=np.float32)
>>> out = np.zeros((2, 3), dtype=np.float32)
>>> items = [space.sample() for _ in range(2)]
>>> concatenate(space, items, out)
array([[0.77395606, 0.43887845, 0.85859793],
       [0.697368  , 0.09417735, 0.97562236]], dtype=float32)
```

### gymnasium.vector.utils.iterate(space: [Space](../spaces.md#gymnasium.spaces.Space)[\_T], items: \_T) → Iterator[Any]

### gymnasium.vector.utils.iterate(space: [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete), items: Iterable[Any]) → Never

### gymnasium.vector.utils.iterate(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), items: ndarray) → Iterator[Any]

### gymnasium.vector.utils.iterate(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), items: ndarray) → Iterator[Any]

### gymnasium.vector.utils.iterate(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), items: ndarray) → Iterator[Any]

### gymnasium.vector.utils.iterate(space: [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple), items: tuple[Any, ...]) → Iterator[Any]

### gymnasium.vector.utils.iterate(space: [Dict](../spaces/composite.md#gymnasium.spaces.Dict), items: Mapping[str, Any]) → Iterator[dict[str, Any]]

Iterate over the elements of a (batched) space.

* **Parameters:**
  * **space** – (batched) space (e.g. action_space or observation_space from vectorized environment).
  * **items** – Batched samples to be iterated over (e.g. sample from the space).

### Example

```pycon
>>> from gymnasium.spaces import Box, Dict
>>> import numpy as np
>>> space = Dict({
... 'position': Box(low=0, high=1, shape=(2, 3), seed=42, dtype=np.float32),
... 'velocity': Box(low=0, high=1, shape=(2, 2), seed=42, dtype=np.float32)})
>>> items = space.sample()
>>> it = iterate(space, items)
>>> next(it)
{'position': array([0.77395606, 0.43887845, 0.85859793], dtype=float32), 'velocity': array([0.77395606, 0.43887845], dtype=float32)}
>>> next(it)
{'position': array([0.697368  , 0.09417735, 0.97562236], dtype=float32), 'velocity': array([0.85859793, 0.697368  ], dtype=float32)}
>>> next(it)
Traceback (most recent call last):
    ...
StopIteration
```

### gymnasium.vector.utils.create_empty_array(space: [Space](../spaces.md#gymnasium.spaces.Space), n: int = 1, fn: Callable = np.zeros) → tuple[Any, ...] | dict[str, Any] | ndarray | None

### gymnasium.vector.utils.create_empty_array(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box), n: int = 1, fn: Callable = np.zeros) → ndarray

### gymnasium.vector.utils.create_empty_array(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box), n: int = 1, fn: Callable = np.zeros) → ndarray

### gymnasium.vector.utils.create_empty_array(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box), n: int = 1, fn: Callable = np.zeros) → ndarray

### gymnasium.vector.utils.create_empty_array(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box), n: int = 1, fn: Callable = np.zeros) → ndarray

### gymnasium.vector.utils.create_empty_array(space: [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple), n: int = 1, fn: Callable = np.zeros) → tuple[Any, ...]

### gymnasium.vector.utils.create_empty_array(space: [Dict](../spaces/composite.md#gymnasium.spaces.Dict), n: int = 1, fn: Callable = np.zeros) → dict[str, Any]

### gymnasium.vector.utils.create_empty_array(space: [Graph](../spaces/composite.md#gymnasium.spaces.Graph), n: int = 1, fn: Callable = np.zeros) → tuple[[GraphInstance](../spaces/composite.md#gymnasium.spaces.GraphInstance), ...]

### gymnasium.vector.utils.create_empty_array(space: [Text](../spaces/fundamental.md#gymnasium.spaces.Text), n: int = 1, fn: Callable = np.zeros) → tuple[str, ...]

### gymnasium.vector.utils.create_empty_array(space: [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence), n: int = 1, fn: Callable = np.zeros) → tuple[Any, ...]

### gymnasium.vector.utils.create_empty_array(space: [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf), n: int = 1, fn: Callable = np.zeros) → tuple[tuple[()], ...]

### gymnasium.vector.utils.create_empty_array(space: [Space](../spaces.md#gymnasium.spaces.Space), n: int = 1, fn: Callable = np.zeros) → None

Create an empty (possibly nested and normally numpy-based) array, used in conjunction with `concatenate(..., out=array)`.

In most cases, the array will be contained within the batched space, however, this is not guaranteed.

* **Parameters:**
  * **space** – Observation space of a single environment in the vectorized environment.
  * **n** – Number of environments in the vectorized environment. If `None`, creates an empty sample from `space`.
  * **fn** – Function to apply when creating the empty numpy array. Examples of such functions are `np.empty` or `np.zeros`.
* **Returns:**
  **The output object. This object is a** (*possibly nested*)
* **Raises:**
  **ValueError** – Space is not a valid `gymnasium.Space` instance

### Example

```pycon
>>> from gymnasium.spaces import Box, Dict
>>> import numpy as np
>>> space = Dict({
...     'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
...     'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32),
... })
>>> create_empty_array(space, n=2, fn=np.zeros)
{'position': array([[0., 0., 0.],
       [0., 0., 0.]], dtype=float32), 'velocity': array([[0., 0.],
       [0., 0.]], dtype=float32)}
```

## Shared Memory for a Space

### gymnasium.vector.utils.create_shared_memory(space: [Space](../spaces.md#gymnasium.spaces.Space)[Any], n: int = 1, ctx: ModuleType = mp) → dict[str, Any] | tuple[Any, ...] | SynchronizedArray

### gymnasium.vector.utils.create_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), n: int = 1, ctx: ModuleType = mp) → SynchronizedArray[Any]

### gymnasium.vector.utils.create_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), n: int = 1, ctx: ModuleType = mp) → SynchronizedArray[Any]

### gymnasium.vector.utils.create_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), n: int = 1, ctx: ModuleType = mp) → SynchronizedArray[Any]

### gymnasium.vector.utils.create_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), n: int = 1, ctx: ModuleType = mp) → SynchronizedArray[Any]

### gymnasium.vector.utils.create_shared_memory(space: [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple), n: int = 1, ctx: ModuleType = mp) → tuple[Any, ...]

### gymnasium.vector.utils.create_shared_memory(space: [Dict](../spaces/composite.md#gymnasium.spaces.Dict), n: int = 1, ctx: ModuleType = mp) → dict[str, Any]

### gymnasium.vector.utils.create_shared_memory(space: [Text](../spaces/fundamental.md#gymnasium.spaces.Text), n: int = 1, ctx: ModuleType = mp) → SynchronizedArray[c_int32]

### gymnasium.vector.utils.create_shared_memory(space: [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf), n: int = 1, ctx: ModuleType = mp) → \_SharedMemoryOneOf

### gymnasium.vector.utils.create_shared_memory(space: [Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence), n: int = 1, ctx: ModuleType = mp) → Never

### gymnasium.vector.utils.create_shared_memory(space: [Graph](../spaces/composite.md#gymnasium.spaces.Graph) | [Sequence](../spaces/composite.md#gymnasium.spaces.Sequence), n: int = 1, ctx: ModuleType = mp) → Never

Create a shared memory object, to be shared across processes.

This eventually contains the observations from the vectorized environment.

* **Parameters:**
  * **space** – Observation space of a single environment in the vectorized environment.
  * **n** – Number of environments in the vectorized environment (i.e. the number of processes).
  * **ctx** – The multiprocess module
* **Returns:**
  **shared_memory for the shared object across processes.**
* **Raises:**
  [**CustomSpaceError**](../utils.md#gymnasium.error.CustomSpaceError) – Space is not a valid `gymnasium.Space` instance

### gymnasium.vector.utils.read_from_shared_memory(space: [Space](../spaces.md#gymnasium.spaces.Space), shared_memory: dict[str, Any] | tuple[Any, ...] | SynchronizedArray, n: int = 1) → dict[str, Any] | tuple[Any, ...] | ndarray

### gymnasium.vector.utils.read_from_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), shared_memory: SynchronizedArray[Any], n: int = 1) → np.ndarray

### gymnasium.vector.utils.read_from_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), shared_memory: SynchronizedArray[Any], n: int = 1) → np.ndarray

### gymnasium.vector.utils.read_from_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), shared_memory: SynchronizedArray[Any], n: int = 1) → np.ndarray

### gymnasium.vector.utils.read_from_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), shared_memory: SynchronizedArray[Any], n: int = 1) → np.ndarray

### gymnasium.vector.utils.read_from_shared_memory(space: [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple), shared_memory: tuple[dict[str, Any] | tuple[Any, ...] | SynchronizedArray, ...], n: int = 1) → tuple[Any, ...]

### gymnasium.vector.utils.read_from_shared_memory(space: [Dict](../spaces/composite.md#gymnasium.spaces.Dict), shared_memory: dict[str, dict[str, Any] | tuple[Any, ...] | SynchronizedArray], n: int = 1) → dict[str, Any]

### gymnasium.vector.utils.read_from_shared_memory(space: [Text](../spaces/fundamental.md#gymnasium.spaces.Text), shared_memory: SynchronizedArray[c_int32], n: int = 1) → tuple[str, ...]

### gymnasium.vector.utils.read_from_shared_memory(space: [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf), shared_memory: \_SharedMemoryOneOf, n: int = 1) → tuple[Any, ...]

Read the batch of observations from shared memory as a numpy array.

..notes::
: The numpy array objects returned by read_from_shared_memory shares the
  memory of shared_memory. Any changes to shared_memory are forwarded
  to observations, and vice-versa. To avoid any side-effect, use np.copy.

* **Parameters:**
  * **space** – Observation space of a single environment in the vectorized environment.
  * **shared_memory** – Shared object across processes. This contains the observations from the vectorized environment.
    This object is created with create_shared_memory.
  * **n** – Number of environments in the vectorized environment (i.e. the number of processes).
* **Returns:**
  **Batch of observations as a** (*possibly nested*)
* **Raises:**
  [**CustomSpaceError**](../utils.md#gymnasium.error.CustomSpaceError) – Space is not a valid `gymnasium.Space` instance

### gymnasium.vector.utils.write_to_shared_memory(space: [Space](../spaces.md#gymnasium.spaces.Space), index: int, value: ndarray, shared_memory: dict[str, Any] | tuple[Any, ...] | SynchronizedArray) → None

### gymnasium.vector.utils.write_to_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), index: int, value: np.typing.ArrayLike, shared_memory: SynchronizedArray[Any]) → None

### gymnasium.vector.utils.write_to_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), index: int, value: np.typing.ArrayLike, shared_memory: SynchronizedArray[Any]) → None

### gymnasium.vector.utils.write_to_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), index: int, value: np.typing.ArrayLike, shared_memory: SynchronizedArray[Any]) → None

### gymnasium.vector.utils.write_to_shared_memory(space: [Box](../spaces/fundamental.md#gymnasium.spaces.Box) | [Discrete](../spaces/fundamental.md#gymnasium.spaces.Discrete) | [MultiDiscrete](../spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) | [MultiBinary](../spaces/fundamental.md#gymnasium.spaces.MultiBinary), index: int, value: np.typing.ArrayLike, shared_memory: SynchronizedArray[Any]) → None

### gymnasium.vector.utils.write_to_shared_memory(space: [Tuple](../spaces/composite.md#gymnasium.spaces.Tuple), index: int, values: tuple[Any, ...], shared_memory: tuple[dict[str, Any] | tuple[Any, ...] | SynchronizedArray, ...]) → None

### gymnasium.vector.utils.write_to_shared_memory(space: [Dict](../spaces/composite.md#gymnasium.spaces.Dict), index: int, values: dict[str, Any], shared_memory: Mapping[str, Any]) → None

### gymnasium.vector.utils.write_to_shared_memory(space: [Text](../spaces/fundamental.md#gymnasium.spaces.Text), index: int, values: str, shared_memory: SynchronizedArray[c_int32]) → None

### gymnasium.vector.utils.write_to_shared_memory(space: [OneOf](../spaces/composite.md#gymnasium.spaces.OneOf), index: int, values: tuple[int, Any], shared_memory: \_SharedMemoryOneOf) → None

Write the observation of a single environment into shared memory.

* **Parameters:**
  * **space** – Observation space of a single environment in the vectorized environment.
  * **index** – Index of the environment (must be in [0, num_envs)).
  * **value** – Observation of the single environment to write to shared memory.
  * **shared_memory** – Shared object across processes. This contains the observations from the vectorized environment.
    This object is created with create_shared_memory.
* **Raises:**
  [**CustomSpaceError**](../utils.md#gymnasium.error.CustomSpaceError) – Space is not a valid `gymnasium.Space` instance

## Miscellaneous

### *class* gymnasium.vector.utils.CloudpickleWrapper(fn: Callable[[], \_EnvT_co])

Wrapper that uses cloudpickle to pickle and unpickle the result.

Cloudpickle wrapper for a function.

### gymnasium.vector.utils.clear_mpi_env_vars() → Generator[None, None, None]

Clears the MPI of environment variables.

`from mpi4py import MPI` will call `MPI_Init` by default.
If the child process has MPI environment variables, MPI will think that the child process
is an MPI process just like the parent and do bad things such as hang.

This context manager is a hacky way to clear those environment variables
temporarily such as when we are starting multiprocessing Processes.

* **Yields:**
  Yields for the context manager
