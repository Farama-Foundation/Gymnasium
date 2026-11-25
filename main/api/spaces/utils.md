# Spaces Utils

### gymnasium.spaces.utils.flatten_space(space: [Space](../spaces.md#gymnasium.spaces.Space)[Any]) → [Box](fundamental.md#gymnasium.spaces.Box) | [Dict](composite.md#gymnasium.spaces.Dict) | [Sequence](composite.md#gymnasium.spaces.Sequence) | [Tuple](composite.md#gymnasium.spaces.Tuple) | [Graph](composite.md#gymnasium.spaces.Graph)

### gymnasium.spaces.utils.flatten_space(space: [Box](fundamental.md#gymnasium.spaces.Box)) → [Box](fundamental.md#gymnasium.spaces.Box)

### gymnasium.spaces.utils.flatten_space(space: [Discrete](fundamental.md#gymnasium.spaces.Discrete) | [MultiBinary](fundamental.md#gymnasium.spaces.MultiBinary) | [MultiDiscrete](fundamental.md#gymnasium.spaces.MultiDiscrete)) → [Box](fundamental.md#gymnasium.spaces.Box)

### gymnasium.spaces.utils.flatten_space(space: [Discrete](fundamental.md#gymnasium.spaces.Discrete) | [MultiBinary](fundamental.md#gymnasium.spaces.MultiBinary) | [MultiDiscrete](fundamental.md#gymnasium.spaces.MultiDiscrete)) → [Box](fundamental.md#gymnasium.spaces.Box)

### gymnasium.spaces.utils.flatten_space(space: [Discrete](fundamental.md#gymnasium.spaces.Discrete) | [MultiBinary](fundamental.md#gymnasium.spaces.MultiBinary) | [MultiDiscrete](fundamental.md#gymnasium.spaces.MultiDiscrete)) → [Box](fundamental.md#gymnasium.spaces.Box)

### gymnasium.spaces.utils.flatten_space(space: [Tuple](composite.md#gymnasium.spaces.Tuple)) → [Box](fundamental.md#gymnasium.spaces.Box) | [Tuple](composite.md#gymnasium.spaces.Tuple)

### gymnasium.spaces.utils.flatten_space(space: [Dict](composite.md#gymnasium.spaces.Dict)) → [Box](fundamental.md#gymnasium.spaces.Box) | [Dict](composite.md#gymnasium.spaces.Dict)

### gymnasium.spaces.utils.flatten_space(space: [Graph](composite.md#gymnasium.spaces.Graph)) → [Graph](composite.md#gymnasium.spaces.Graph)

### gymnasium.spaces.utils.flatten_space(space: [Text](fundamental.md#gymnasium.spaces.Text)) → [Box](fundamental.md#gymnasium.spaces.Box)

### gymnasium.spaces.utils.flatten_space(space: [Sequence](composite.md#gymnasium.spaces.Sequence)) → [Sequence](composite.md#gymnasium.spaces.Sequence)

### gymnasium.spaces.utils.flatten_space(space: [OneOf](composite.md#gymnasium.spaces.OneOf)) → [Box](fundamental.md#gymnasium.spaces.Box)

Flatten a space into a space that is as flat as possible.

This function will attempt to flatten `space` into a single [`gymnasium.spaces.Box`](fundamental.md#gymnasium.spaces.Box) space.
However, this might not be possible when `space` is an instance of [`gymnasium.spaces.Graph`](composite.md#gymnasium.spaces.Graph),
[`gymnasium.spaces.Sequence`](composite.md#gymnasium.spaces.Sequence) or a compound space that contains a [`gymnasium.spaces.Graph`](composite.md#gymnasium.spaces.Graph)
or [`gymnasium.spaces.Sequence`](composite.md#gymnasium.spaces.Sequence) space.
This is equivalent to [`flatten()`](#gymnasium.spaces.utils.flatten), but operates on the space itself. The
result for non-graph spaces is always a [`gymnasium.spaces.Box`](fundamental.md#gymnasium.spaces.Box) with flat boundaries. While
the result for graph spaces is always a [`gymnasium.spaces.Graph`](composite.md#gymnasium.spaces.Graph) with
`Graph.node_space` being a `Box`
with flat boundaries and `Graph.edge_space` being a `Box` with flat boundaries or
`None`. The box has exactly [`flatdim()`](#gymnasium.spaces.utils.flatdim) dimensions. Flattening a sample
of the original space has the same effect as taking a sample of the flattened
space. However, sampling from the flattened space is not necessarily reversible.
For example, sampling from a flattened Discrete space is the same as sampling from
a Box, and the results may not be integers or one-hot encodings. This may result in
errors or non-uniform sampling.

* **Parameters:**
  **space** – The space to flatten
* **Returns:**
  **A flattened Box**
* **Raises:**
  **NotImplementedError** – if the space is not defined in [`gymnasium.spaces`](../spaces.md#module-gymnasium.spaces).

Example - Flatten spaces.Box:
: ```pycon
  >>> from gymnasium.spaces import Box
  >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
  >>> box
  Box(0.0, 1.0, (3, 4, 5), float32)
  >>> flatten_space(box)
  Box(0.0, 1.0, (60,), float32)
  >>> flatten(box, box.sample()) in flatten_space(box)
  True
  ```

Example - Flatten spaces.Discrete:
: ```pycon
  >>> from gymnasium.spaces import Discrete
  >>> discrete = Discrete(5)
  >>> flatten_space(discrete)
  Box(0, 1, (5,), int64)
  >>> flatten(discrete, discrete.sample()) in flatten_space(discrete)
  True
  ```

Example - Flatten spaces.Dict:
: ```pycon
  >>> from gymnasium.spaces import Dict, Discrete, Box
  >>> space = Dict({"position": Discrete(2), "velocity": Box(0, 1, shape=(2, 2))})
  >>> flatten_space(space)
  Box(0.0, 1.0, (6,), float64)
  >>> flatten(space, space.sample()) in flatten_space(space)
  True
  ```

Example - Flatten spaces.Graph:
: ```pycon
  >>> from gymnasium.spaces import Graph, Discrete, Box
  >>> space = Graph(node_space=Box(low=-100, high=100, shape=(3, 4)), edge_space=Discrete(5))
  >>> flatten_space(space)
  Graph(Box(-100.0, 100.0, (12,), float32), Box(0, 1, (5,), int64))
  >>> flatten(space, space.sample()) in flatten_space(space)
  True
  ```

### gymnasium.spaces.utils.flatten(space: [Space](../spaces.md#gymnasium.spaces.Space)[T], x: T) → NDArray[Any] | dict[str, Any] | tuple[Any, ...] | [GraphInstance](composite.md#gymnasium.spaces.GraphInstance)

### gymnasium.spaces.utils.flatten(space: [Box](fundamental.md#gymnasium.spaces.Box) | [MultiBinary](fundamental.md#gymnasium.spaces.MultiBinary), x: NDArray[Any]) → NDArray[Any]

### gymnasium.spaces.utils.flatten(space: [Box](fundamental.md#gymnasium.spaces.Box) | [MultiBinary](fundamental.md#gymnasium.spaces.MultiBinary), x: NDArray[Any]) → NDArray[Any]

### gymnasium.spaces.utils.flatten(space: [Discrete](fundamental.md#gymnasium.spaces.Discrete), x: \_IntegerT) → NDArray[\_IntegerT]

### gymnasium.spaces.utils.flatten(space: [MultiDiscrete](fundamental.md#gymnasium.spaces.MultiDiscrete), x: NDArray[int64]) → NDArray[int64]

### gymnasium.spaces.utils.flatten(space: [Tuple](composite.md#gymnasium.spaces.Tuple), x: tuple[Any, ...]) → tuple[Any, ...] | NDArray[Any]

### gymnasium.spaces.utils.flatten(space: [Dict](composite.md#gymnasium.spaces.Dict), x: dict[str, Any]) → dict[str, Any] | NDArray[Any]

### gymnasium.spaces.utils.flatten(space: [Graph](composite.md#gymnasium.spaces.Graph), x: [GraphInstance](composite.md#gymnasium.spaces.GraphInstance)) → [GraphInstance](composite.md#gymnasium.spaces.GraphInstance)

### gymnasium.spaces.utils.flatten(space: [Text](fundamental.md#gymnasium.spaces.Text), x: str) → NDArray[int32]

### gymnasium.spaces.utils.flatten(space: [Sequence](composite.md#gymnasium.spaces.Sequence), x: tuple[Any, ...] | Any) → tuple[Any, ...] | Any

### gymnasium.spaces.utils.flatten(space: [OneOf](composite.md#gymnasium.spaces.OneOf), x: tuple[int, Any]) → NDArray[Any]

Flatten a data point from a space.

This is useful when e.g. points from spaces must be passed to a neural
network, which only understands flat arrays of floats.

* **Parameters:**
  * **space** – The space that `x` is flattened by
  * **x** – The value to flatten
* **Returns:**
  **The flattened datapoint** –
  - For [`gymnasium.spaces.Box`](fundamental.md#gymnasium.spaces.Box) and [`gymnasium.spaces.MultiBinary`](fundamental.md#gymnasium.spaces.MultiBinary), this is a flattened array
  - For [`gymnasium.spaces.Discrete`](fundamental.md#gymnasium.spaces.Discrete) and [`gymnasium.spaces.MultiDiscrete`](fundamental.md#gymnasium.spaces.MultiDiscrete), this is a flattened one-hot array of the sample
  - For [`gymnasium.spaces.Tuple`](composite.md#gymnasium.spaces.Tuple) and [`gymnasium.spaces.Dict`](composite.md#gymnasium.spaces.Dict), this is a concatenated array the subspaces (does not support graph subspaces)
  - For graph spaces, returns `GraphInstance` where:
    : - `GraphInstance.nodes` are n x k arrays
      - `GraphInstance.edges` are either:
        : - m x k arrays
          - None
      - `GraphInstance.edge_links` are either:
        : - m x 2 arrays
          - None
* **Raises:**
  **NotImplementedError** – If the space is not defined in [`gymnasium.spaces`](../spaces.md#module-gymnasium.spaces).

### Example

```pycon
>>> from gymnasium.spaces import Box, Discrete, Tuple
>>> space = Box(0, 1, shape=(3, 5))
>>> flatten(space, space.sample()).shape
(15,)
>>> space = Discrete(4)
>>> flatten(space, 2)
array([0, 0, 1, 0])
>>> space = Tuple((Box(0, 1, shape=(2,)), Box(0, 1, shape=(3,)), Discrete(3)))
>>> example = ((.5, .25), (1., 0., .2), 1)
>>> flatten(space, example)
array([0.5 , 0.25, 1.  , 0.  , 0.2 , 0.  , 1.  , 0.  ])
```

### gymnasium.spaces.utils.flatdim(space: [Space](../spaces.md#gymnasium.spaces.Space)[Any]) → int

### gymnasium.spaces.utils.flatdim(space: [Box](fundamental.md#gymnasium.spaces.Box) | [MultiBinary](fundamental.md#gymnasium.spaces.MultiBinary)) → int

### gymnasium.spaces.utils.flatdim(space: [Box](fundamental.md#gymnasium.spaces.Box) | [MultiBinary](fundamental.md#gymnasium.spaces.MultiBinary)) → int

### gymnasium.spaces.utils.flatdim(space: [Discrete](fundamental.md#gymnasium.spaces.Discrete)) → int

### gymnasium.spaces.utils.flatdim(space: [MultiDiscrete](fundamental.md#gymnasium.spaces.MultiDiscrete)) → int

### gymnasium.spaces.utils.flatdim(space: [Tuple](composite.md#gymnasium.spaces.Tuple)) → int

### gymnasium.spaces.utils.flatdim(space: [Dict](composite.md#gymnasium.spaces.Dict)) → int

### gymnasium.spaces.utils.flatdim(space: [Graph](composite.md#gymnasium.spaces.Graph))

### gymnasium.spaces.utils.flatdim(space: [Text](fundamental.md#gymnasium.spaces.Text)) → int

### gymnasium.spaces.utils.flatdim(space: [OneOf](composite.md#gymnasium.spaces.OneOf)) → int

Return the number of dimensions a flattened equivalent of this space would have.

* **Parameters:**
  **space** – The space to return the number of dimensions of the flattened spaces
* **Returns:**
  **The number of dimensions for the flattened spaces**
* **Raises:**
  * **NotImplementedError** – if the space is not defined in `gym.spaces`.
  * **ValueError** – if the space cannot be flattened into a [`gymnasium.spaces.Box`](fundamental.md#gymnasium.spaces.Box)

### Example

```pycon
>>> from gymnasium.spaces import Dict, Discrete
>>> space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
>>> flatdim(space)
5
```

### gymnasium.spaces.utils.unflatten(space: [Space](../spaces.md#gymnasium.spaces.Space)[T], x: NDArray[Any] | dict[str, Any] | tuple[Any, ...] | [GraphInstance](composite.md#gymnasium.spaces.GraphInstance)) → T

### gymnasium.spaces.utils.unflatten(space: [Box](fundamental.md#gymnasium.spaces.Box) | [MultiBinary](fundamental.md#gymnasium.spaces.MultiBinary), x: NDArray[Any]) → NDArray[Any]

### gymnasium.spaces.utils.unflatten(space: [Box](fundamental.md#gymnasium.spaces.Box) | [MultiBinary](fundamental.md#gymnasium.spaces.MultiBinary), x: NDArray[Any]) → NDArray[Any]

### gymnasium.spaces.utils.unflatten(space: [Discrete](fundamental.md#gymnasium.spaces.Discrete), x: NDArray[\_IntegerT | float64]) → \_IntegerT

### gymnasium.spaces.utils.unflatten(space: [MultiDiscrete](fundamental.md#gymnasium.spaces.MultiDiscrete), x: NDArray[integer[Any]]) → NDArray[integer[Any]]

### gymnasium.spaces.utils.unflatten(space: [Tuple](composite.md#gymnasium.spaces.Tuple), x: NDArray[Any] | tuple[Any, ...]) → tuple[Any, ...]

### gymnasium.spaces.utils.unflatten(space: [Dict](composite.md#gymnasium.spaces.Dict), x: NDArray[Any] | dict[str, Any]) → dict[str, Any]

### gymnasium.spaces.utils.unflatten(space: [Graph](composite.md#gymnasium.spaces.Graph), x: [GraphInstance](composite.md#gymnasium.spaces.GraphInstance)) → [GraphInstance](composite.md#gymnasium.spaces.GraphInstance)

### gymnasium.spaces.utils.unflatten(space: [Text](fundamental.md#gymnasium.spaces.Text), x: NDArray[int32]) → str

### gymnasium.spaces.utils.unflatten(space: [Sequence](composite.md#gymnasium.spaces.Sequence), x: tuple[Any, ...]) → tuple[Any, ...] | Any

### gymnasium.spaces.utils.unflatten(space: [OneOf](composite.md#gymnasium.spaces.OneOf), x: NDArray[Any]) → tuple[int, Any]

Unflatten a data point from a space.

This reverses the transformation applied by [`flatten()`](#gymnasium.spaces.utils.flatten). You must ensure
that the `space` argument is the same as for the [`flatten()`](#gymnasium.spaces.utils.flatten) call.

* **Parameters:**
  * **space** – The space used to unflatten `x`
  * **x** – The array to unflatten
* **Returns:**
  **A point with a structure that matches the space.**
* **Raises:**
  **NotImplementedError** – if the space is not defined in [`gymnasium.spaces`](../spaces.md#module-gymnasium.spaces).

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [Space](../spaces.md#gymnasium.spaces.Space), space_2: [Space](../spaces.md#gymnasium.spaces.Space)) → bool

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [MultiBinary](fundamental.md#gymnasium.spaces.MultiBinary), space_2)

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [MultiDiscrete](fundamental.md#gymnasium.spaces.MultiDiscrete), space_2)

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [Discrete](fundamental.md#gymnasium.spaces.Discrete), space_2)

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [Box](fundamental.md#gymnasium.spaces.Box), space_2)

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [Text](fundamental.md#gymnasium.spaces.Text), space_2)

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [Dict](composite.md#gymnasium.spaces.Dict), space_2)

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [Tuple](composite.md#gymnasium.spaces.Tuple), space_2)

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [Graph](composite.md#gymnasium.spaces.Graph), space_2)

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [OneOf](composite.md#gymnasium.spaces.OneOf), space_2)

### gymnasium.spaces.utils.is_space_dtype_shape_equiv(space_1: [Sequence](composite.md#gymnasium.spaces.Sequence), space_2)

Returns if two spaces share a common dtype and shape (plus any critical variables).

This function is primarily used to check for compatibility of different spaces in a vector environment.

* **Parameters:**
  * **space_1** – A Gymnasium space
  * **space_2** – A Gymnasium space
* **Returns:**
  **If the two spaces share a common dtype and shape** (*plus any critical variables*)
