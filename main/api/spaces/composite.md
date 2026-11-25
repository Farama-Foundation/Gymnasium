# Composite Spaces

### *class* gymnasium.spaces.Dict(spaces: dict[str, [Space](../spaces.md#gymnasium.spaces.Space)[\_T_co]] | Sequence[tuple[str, [Space](../spaces.md#gymnasium.spaces.Space)[\_T_co]]] | None = None, seed: dict | int | Generator | None = None, \*, sort_keys: bool = True, \*\*spaces_kwargs: [Space](../spaces.md#gymnasium.spaces.Space))

A dictionary of [`Space`](../spaces.md#gymnasium.spaces.Space) instances.

Elements of this space are (ordered) dictionaries of elements from the constituent spaces.

### Example

```pycon
>>> from gymnasium.spaces import Dict, Box, Discrete
>>> observation_space = Dict({"position": Box(-1, 1, shape=(2,)), "color": Discrete(3)}, seed=42)
>>> observation_space.sample()
{'color': np.int64(0), 'position': array([-0.3991573 ,  0.21649833], dtype=float32)}
```

With a nested dict:

```pycon
>>> from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
>>> Dict(  
...     {
...         "ext_controller": MultiDiscrete([5, 2, 2]),
...         "inner_state": Dict(
...             {
...                 "charge": Discrete(100),
...                 "system_checks": MultiBinary(10),
...                 "job_status": Dict(
...                     {
...                         "task": Discrete(5),
...                         "progress": Box(low=0, high=100, shape=()),
...                     }
...                 ),
...             }
...         ),
...     }
... )
```

It can be convenient to use [`Dict`](#gymnasium.spaces.Dict) spaces if you want to make complex observations or actions more human-readable.
Usually, it will not be possible to use elements of this space directly in learning code. However, you can easily
convert [`Dict`](#gymnasium.spaces.Dict) observations to flat arrays by using a [`gymnasium.wrappers.FlattenObservation`](../wrappers/observation_wrappers.md#gymnasium.wrappers.FlattenObservation) wrapper.
Similar wrappers can be implemented to deal with [`Dict`](#gymnasium.spaces.Dict) actions.

* **Parameters:**
  * **spaces** – A dictionary of spaces. This specifies the structure of the [`Dict`](#gymnasium.spaces.Dict) space
  * **seed** – Optionally, you can use this argument to seed the RNGs of the spaces that make up the [`Dict`](#gymnasium.spaces.Dict) space.
  * **sort_keys** – Whether to sort the keys of a standard mapping. This does not affect `OrderedDict` or sequence inputs.
  * **\*\*spaces_kwargs** – If `spaces` is `None`, you need to pass the constituent spaces as keyword arguments, as described above.

#### sample(mask: dict[str, Any] | None = None, probability: dict[str, Any] | None = None) → dict[str, Any]

Generates a single random sample from this space.

The sample is an ordered dictionary of independent samples from the constituent spaces.

* **Parameters:**
  * **mask** – An optional mask for each of the subspaces, expects the same keys as the space
  * **probability** – An optional probability mask for each of the subspaces, expects the same keys as the space
* **Returns:**
  **A dictionary with the same key and sampled values from :attr:\`self.spaces\`**

#### seed(seed: int | dict[str, Any] | None = None) → dict[str, int]

Seed the PRNG of this space and all subspaces.

Depending on the type of seed, the subspaces will be seeded differently

* `None` - All the subspaces will use a random initial seed
* `Int` - The integer is used to seed the [`Dict`](#gymnasium.spaces.Dict) space that is used to generate seed values for each of the subspaces. Warning, this does not guarantee unique seeds for all subspaces, though is very unlikely.
* `Dict` - A dictionary of seeds for each subspace, requires a seed key for every subspace. This supports seeding of multiple composite subspaces (`Dict["space": Dict[...], ...]` with `{"space": {...}, ...}`).

* **Parameters:**
  **seed** – An optional int or dictionary of subspace keys to int to seed each PRNG. See above for more details.
* **Returns:**
  **A dictionary for the seed values of the subspaces**

### *class* gymnasium.spaces.Tuple(spaces: Iterable[[Space](../spaces.md#gymnasium.spaces.Space)[\_T_co]], seed: int | Generator | None = None)

A tuple (more precisely: the cartesian product) of [`Space`](../spaces.md#gymnasium.spaces.Space) instances.

Elements of this space are tuples of elements of the constituent spaces.

### Example

```pycon
>>> from gymnasium.spaces import Tuple, Box, Discrete
>>> observation_space = Tuple((Discrete(2), Box(-1, 1, shape=(2,))), seed=42)
>>> observation_space.sample()
(np.int64(0), array([-0.3991573 ,  0.21649833], dtype=float32))
```

* **Parameters:**
  * **spaces** (*Iterable* *[*[*Space*](../spaces.md#gymnasium.spaces.Space) *]*) – The spaces that are involved in the cartesian product.
  * **seed** – Optionally, you can use this argument to seed the RNGs of the `spaces` to ensure reproducible sampling.

#### sample(mask: tuple[Any | None, ...] | None = None, probability: tuple[Any | None, ...] | None = None) → tuple[\_T_co, ...]

Generates a single random sample inside this space.

This method draws independent samples from the subspaces.

* **Parameters:**
  * **mask** – An optional tuple of optional masks for each of the subspace’s samples,
    expects the same number of masks as spaces
  * **probability** – An optional tuple of optional probability masks for each of the subspace’s samples,
    expects the same number of probability masks as spaces
* **Returns:**
  **Tuple of the subspace’s samples**

#### seed(seed: int | Sequence[int] | None = None) → tuple[int, ...]

Seed the PRNG of this space and all subspaces.

Depending on the type of seed, the subspaces will be seeded differently

* `None` - All the subspaces will use a random initial seed
* `Int` - The integer is used to seed the [`Tuple`](#gymnasium.spaces.Tuple) space that is used to generate seed values for each of the subspaces. Warning, this does not guarantee unique seeds for all the subspaces.
* `List` / `Tuple` - Values used to seed the subspaces. This allows the seeding of multiple composite subspaces `[42, 54, ...]`.

* **Parameters:**
  **seed** – An optional list of ints or int to seed the (sub-)spaces.
* **Returns:**
  **A tuple of the seed values for all subspaces**

### *class* gymnasium.spaces.Sequence(space: [Space](../spaces.md#gymnasium.spaces.Space)[Any], seed: int | Generator | None = None, stack: bool = False)

This space represent sets of finite-length sequences.

This space represents the set of tuples of the form $(a_0, \dots, a_n)$ where the $a_i$ belong
to some space that is specified during initialization and the integer $n$ is not fixed

### Example

```pycon
>>> from gymnasium.spaces import Sequence, Box
>>> observation_space = Sequence(Box(0, 1), seed=0)
>>> observation_space.sample()
(array([0.6822636], dtype=float32), array([0.18933342], dtype=float32), array([0.19049619], dtype=float32))
>>> observation_space.sample()
(array([0.83506], dtype=float32), array([0.9053838], dtype=float32), array([0.5836242], dtype=float32), array([0.63214064], dtype=float32))
```

Example with stacked observations
: ```pycon
  >>> observation_space = Sequence(Box(0, 1), stack=True, seed=0)
  >>> observation_space.sample()
  array([[0.6822636 ],
         [0.18933342],
         [0.19049619]], dtype=float32)
  ```

* **Parameters:**
  * **space** – Elements in the sequences this space represent must belong to this space.
  * **seed** – Optionally, you can use this argument to seed the RNG that is used to sample from the space.
  * **stack** – If `True` then the resulting samples would be stacked.

#### sample(mask: tuple[None | int | NDArray[integer], Any] | None = None, probability: tuple[None | int | NDArray[integer], Any] | None = None) → tuple[Any, ...] | Any

Generates a single random sample from this space.

* **Parameters:**
  * **mask** – 

    An optional mask for (optionally) the length of the sequence and (optionally) the values in the sequence.
    If you specify `mask`, it is expected to be a tuple of the form `(length_mask, sample_mask)` where `length_mask` is
    * `None` - The length will be randomly drawn from a geometric distribution
    * `int` - Fixed length
    * `np.ndarray` of integers - Length of the sampled sequence is randomly drawn from this array.

    The second element of the tuple `sample_mask` specifies how the feature space will be sampled.
    Depending on if mask or probability is used will affect what argument is used.
  * **probability** – See mask description above, the only difference is on the `sample_mask` for the feature space being probability rather than mask.
* **Returns:**
  **A tuple of random length with random samples of elements from the :attr:\`feature_space\`.**

#### seed(seed: int | tuple[int, int] | None = None) → tuple[int, int]

Seed the PRNG of the Sequence space and the feature space.

Depending on the type of seed, the subspaces will be seeded differently

* `None` - All the subspaces will use a random initial seed
* `Int` - The integer is used to seed the [`Sequence`](#gymnasium.spaces.Sequence) space that is used to generate a seed value for the feature space.
* `Tuple of ints` - A tuple for the [`Sequence`](#gymnasium.spaces.Sequence) and feature space.

* **Parameters:**
  **seed** – An optional int or tuple of ints to seed the PRNG. See above for more details
* **Returns:**
  **A tuple of the seeding values for the Sequence and feature space**

### *class* gymnasium.spaces.Graph(node_space: [Space](../spaces.md#gymnasium.spaces.Space)[Any], edge_space: None | [Space](../spaces.md#gymnasium.spaces.Space)[Any], seed: int | Generator | None = None)

A space representing graph information as a series of `nodes` connected with `edges` according to an adjacency matrix represented as a series of `edge_links`.

### Example

```pycon
>>> from gymnasium.spaces import Graph, Box, Discrete
>>> observation_space = Graph(node_space=Box(low=-100, high=100, shape=(3,)), edge_space=Discrete(3), seed=123)
>>> observation_space.sample(num_nodes=4, num_edges=8)
GraphInstance(nodes=array([[-50.143734, -89.37025 , -42.120003],
       [ 31.519672, -96.5817  ,  85.16244 ],
       [ 97.65833 ,  92.20077 ,  75.31062 ],
       [-21.80443 , -73.790115,  39.97857 ]], dtype=float32), edges=array([1, 0, 2, 0, 2, 1, 1, 2]), edge_links=array([[0, 2],
       [2, 0],
       [3, 0],
       [1, 0],
       [1, 0],
       [1, 3],
       [1, 3],
       [1, 1]], dtype=int32))
```

* **Parameters:**
  * **node_space** ([*Space*](../spaces.md#gymnasium.spaces.Space) *[**Any* *]*) – space of the node features.
  * **edge_space** (*None* *|* [*Space*](../spaces.md#gymnasium.spaces.Space) *[**Any* *]*) – space of the edge features.
  * **seed** – Optionally, you can use this argument to seed the RNG that is used to sample from the space.

#### sample(mask: tuple[NDArray[Any] | tuple[Any, ...] | None, NDArray[Any] | tuple[Any, ...] | None] | None = None, probability: tuple[NDArray[Any] | tuple[Any, ...] | None, NDArray[Any] | tuple[Any, ...] | None] | None = None, num_nodes: int = 10, num_edges: int | None = None) → [GraphInstance](#gymnasium.spaces.GraphInstance)

Generates a single sample graph with num_nodes between `1` and `10` sampled from the Graph.

* **Parameters:**
  * **mask** – An optional tuple of optional node and edge mask
    (Box spaces don’t support sample masks).
    If no `num_edges` is provided then the `edge_mask` is multiplied by the number of edges
  * **probability** – An optional tuple of optional node and edge probability mask
    (Box spaces don’t support sample probability masks).
    If no `num_edges` is provided then the `edge_mask` is multiplied by the number of edges
  * **num_nodes** – The number of nodes that will be sampled, the default is 10 nodes
  * **num_edges** – An optional number of edges, otherwise, a random number between 0 and $num_nodes^2$
* **Returns:**
  **A :class:\`GraphInstance\` with attributes \`.nodes\`, \`.edges\`, and \`.edge_links\`.**

#### seed(seed: int | tuple[int, int] | tuple[int, int, int] | None = None) → tuple[int, int] | tuple[int, int, int]

Seeds the PRNG of this space and node / edge subspace.

Depending on the type of seed, the subspaces will be seeded differently

* `None` - The root, node and edge spaces PRNG are randomly initialized
* `Int` - The integer is used to seed the [`Graph`](#gymnasium.spaces.Graph) space that is used to generate seed values for the node and edge subspaces.
* `Tuple[int, int]` - Seeds the [`Graph`](#gymnasium.spaces.Graph) and node subspace with a particular value. Only if edge subspace isn’t specified
* `Tuple[int, int, int]` - Seeds the [`Graph`](#gymnasium.spaces.Graph), node and edge subspaces with a particular value.

* **Parameters:**
  **seed** – An optional int or tuple of ints for this space and the node / edge subspaces. See above for more details.
* **Returns:**
  **A tuple of two or three ints depending on if the edge subspace is specified.**

### *class* gymnasium.spaces.GraphInstance(nodes: Iterable[Any], edges: Iterable[Any] | None, edge_links: NDArray[Any] | None)

A Graph space instance.

* nodes (Iterable): an (n x …) sized array representing the features for n nodes, (…) must adhere to the shape of the node space.
* edges (Optional[Iterable]): an (m x …) sized array representing the features for m edges, (…) must adhere to the shape of the edge space.
* edge_links (Optional[np.ndarray]): an (m x 2) sized array of ints representing the indices of the two nodes that each edge connects.

Create new instance of GraphInstance(nodes, edges, edge_links)

### *class* gymnasium.spaces.OneOf(spaces: Iterable[[Space](../spaces.md#gymnasium.spaces.Space)[\_T_co]], seed: int | Generator | None = None)

An exclusive tuple (more precisely: the direct sum) of [`Space`](../spaces.md#gymnasium.spaces.Space) instances.

Elements of this space are elements of one of the constituent spaces.

### Example

```pycon
>>> from gymnasium.spaces import OneOf, Box, Discrete
>>> observation_space = OneOf((Discrete(2), Box(-1, 1, shape=(2,))), seed=123)
>>> observation_space.sample()  # the first element is the space index (Discrete in this case) and the second element is the sample from Discrete
(np.int64(0), np.int64(0))
>>> observation_space.sample()  # this time the Box space was sampled as index=1
(np.int64(1), array([-0.00711833, -0.7257502 ], dtype=float32))
>>> observation_space[0]
Discrete(2)
>>> observation_space[1]
Box(-1.0, 1.0, (2,), float32)
>>> len(observation_space)
2
```

* **Parameters:**
  * **spaces** (*Iterable* *[*[*Space*](../spaces.md#gymnasium.spaces.Space) *]*) – The spaces that are involved in the cartesian product.
  * **seed** – Optionally, you can use this argument to seed the RNGs of the `spaces` to ensure reproducible sampling.

#### sample(mask: tuple[Any | None, ...] | None = None, probability: tuple[Any | None, ...] | None = None) → tuple[int, \_T_co]

Generates a single random sample inside this space.

This method draws independent samples from the subspaces.

* **Parameters:**
  * **mask** – An optional tuple of optional masks for each of the subspace’s samples,
    expects the same number of masks as spaces
  * **probability** – An optional tuple of optional probability masks for each of the subspace’s samples,
    expects the same number of probability masks as spaces
* **Returns:**
  **Tuple of the subspace’s samples**

#### seed(seed: int | tuple[int, ...] | None = None) → tuple[int, ...]

Seed the PRNG of this space and all subspaces.

Depending on the type of seed, the subspaces will be seeded differently

* `None` - All the subspaces will use a random initial seed
* `Int` - The integer is used to seed the [`Tuple`](#gymnasium.spaces.Tuple) space that is used to generate seed values for each of the subspaces. Warning, this does not guarantee unique seeds for all the subspaces.
* `Tuple[int, ...]` - Values used to seed the subspaces, first value seeds the OneOf and subsequent seed the subspaces. This allows the seeding of multiple composite subspaces `[42, 54, ...]`.

* **Parameters:**
  **seed** – An optional int or tuple of ints to seed the OneOf space and subspaces. See above for more details.
* **Returns:**
  **A tuple of ints used to seed the OneOf space and subspaces**
