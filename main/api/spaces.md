# Spaces

<a id="module-gymnasium.spaces"></a>

This module implements various spaces.

Spaces describe mathematical sets and are used in Gym to specify valid actions and observations.
Every Gym environment must have the attributes `action_space` and `observation_space`.
If, for instance, three possible actions (0,1,2) can be performed in your environment and observations
are vectors in the two-dimensional unit cube, the environment code may contain the following two lines:

```default
self.action_space = spaces.Discrete(3)
self.observation_space = spaces.Box(0, 1, shape=(2,))
```

All spaces inherit from the [`Space`](#gymnasium.spaces.Space) superclass.

### *class* gymnasium.spaces.Space(shape: Sequence[int] | None = None, dtype: DTypeLike | None = None, seed: int | Generator | None = None)

Superclass that is used to define observation and action spaces.

Spaces are crucially used in Gym to define the format of valid actions and observations.
They serve various purposes:

* They clearly define how to interact with environments, i.e. they specify what actions need to look like
  and what observations will look like
* They allow us to work with highly structured data (e.g. in the form of elements of [`Dict`](spaces/composite.md#gymnasium.spaces.Dict) spaces)
  and painlessly transform them into flat arrays that can be used in learning code
* They provide a method to sample random elements. This is especially useful for exploration and debugging.

Different spaces can be combined hierarchically via container spaces ([`Tuple`](spaces/composite.md#gymnasium.spaces.Tuple) and [`Dict`](spaces/composite.md#gymnasium.spaces.Dict)) to build a
more expressive space

#### WARNING
Custom observation & action spaces can inherit from the `Space`
class. However, most use-cases should be covered by the existing space
classes (e.g. [`Box`](spaces/fundamental.md#gymnasium.spaces.Box), [`Discrete`](spaces/fundamental.md#gymnasium.spaces.Discrete), etc…), and container classes ([`Tuple`](spaces/composite.md#gymnasium.spaces.Tuple) &
[`Dict`](spaces/composite.md#gymnasium.spaces.Dict)). Note that parametrized probability distributions (through the
[`Space.sample()`](#gymnasium.spaces.Space.sample) method), and batching functions (in `gym.vector.VectorEnv`), are
only well-defined for instances of spaces provided in gym by default.
Moreover, some implementations of Reinforcement Learning algorithms might
not handle custom spaces properly. Use custom spaces with care.

* **Parameters:**
  * **shape** (*Optional* *[*[*Sequence*](spaces/composite.md#gymnasium.spaces.Sequence) *[**int* *]* *]*) – If elements of the space are numpy arrays, this should specify their shape.
  * **dtype** (*Optional* *[**Type* *|* *str* *]*) – If elements of the space are numpy arrays, this should specify their dtype.
  * **seed** – Optionally, you can use this argument to seed the RNG that is used to sample from the space

## Attributes

#### *property* Space.shape *: tuple[int, ...] | None*

Return the shape of the space as an immutable property.

#### *property* Space.dtype

Return the data type of this space.

#### *property* Space.is_np_flattenable *: bool*

Checks whether this space can be flattened to a [`gymnasium.spaces.Box`](spaces/fundamental.md#gymnasium.spaces.Box).

#### *property* Space.np_random *: Generator*

Lazily seed the PRNG since this is expensive and only needed if sampling from this space.

As [`seed()`](#gymnasium.spaces.Space.seed) is not guaranteed to set the \_np_random for particular seeds. We add a
check after [`seed()`](#gymnasium.spaces.Space.seed) to set a new random number generator.

## Methods

Each space implements the following functions:

#### Space.sample(mask: Any | None = None, probability: Any | None = None) → \_T_co

Randomly sample an element of this space.

Can be uniform or non-uniform sampling based on boundedness of space.

The binary mask and the probability mask can’t be used at the same time.

* **Parameters:**
  * **mask** – A mask used for random sampling, expected `dtype=np.int8` and see sample implementation for expected shape.
  * **probability** – A probability mask used for sampling according to the given probability distribution, expected `dtype=np.float64` and see sample implementation for expected shape.
* **Returns:**
  **A sampled actions from the space**

#### Space.contains(x: Any) → bool

Return boolean specifying if x is a valid member of this space, equivalent to `sample in space`.

#### Space.seed(seed: int | None = None) → int | Any

Seed the pseudorandom number generator (PRNG) of this space and, if applicable, the PRNGs of subspaces.

* **Parameters:**
  **seed** – The seed value for the space. This is expanded for composite spaces to accept multiple values. For further details, please refer to the space’s documentation.
* **Returns:**
  **The seed values used for all the PRNGs, for composite spaces this can be a tuple or dictionary of values.**

#### Space.to_jsonable(sample_n: Iterable[\_T_co]) → list[Any] | Any

Convert a batch of samples from this space to a JSONable data type.

#### Space.from_jsonable(sample_n: list[Any]) → list[Any]

Convert a JSONable data type to a batch of samples from this space.

## Fundamental spaces

Gymnasium has a number of fundamental spaces that are used as building blocks for more complex spaces.

* [`Box`](spaces/fundamental.md#gymnasium.spaces.Box) - Supports continuous (and discrete) vectors or matrices, used for vector observations, images, etc
* [`Discrete`](spaces/fundamental.md#gymnasium.spaces.Discrete) - Supports a single discrete number of values with an optional start for the values
* [`MultiBinary`](spaces/fundamental.md#gymnasium.spaces.MultiBinary) - Supports single or matrices of binary values, used for holding down a button or if an agent has an object
* [`MultiDiscrete`](spaces/fundamental.md#gymnasium.spaces.MultiDiscrete) - Supports multiple discrete values with multiple axes, used for controller actions
* [`Text`](spaces/fundamental.md#gymnasium.spaces.Text) - Supports strings, used for passing agent messages, mission details, etc

## Composite spaces

Often environment spaces require joining fundamental spaces together for vectorised environments, separate agents or readability of the space.

* [`Dict`](spaces/composite.md#gymnasium.spaces.Dict) - Supports a dictionary of keys and subspaces, used for a fixed number of unordered spaces
* [`Tuple`](spaces/composite.md#gymnasium.spaces.Tuple) - Supports a tuple of subspaces, used for multiple for a fixed number of ordered spaces
* [`Sequence`](spaces/composite.md#gymnasium.spaces.Sequence) - Supports a variable number of instances of a single subspace, used for entities spaces or selecting a variable number of actions
* [`Graph`](spaces/composite.md#gymnasium.spaces.Graph) - Supports graph based actions or observations with discrete or continuous nodes and edge values
* [`OneOf`](spaces/composite.md#gymnasium.spaces.OneOf) - Supports optional action spaces such that an action can be one of N possible subspaces

## Utility functions

Gymnasium contains a number of helpful utility functions for flattening and unflattening spaces.
This can be important for passing information to neural networks.

* [`utils.flatdim()`](spaces/utils.md#gymnasium.spaces.utils.flatdim) - The number of dimensions the flattened space will contain
* [`utils.flatten_space()`](spaces/utils.md#gymnasium.spaces.utils.flatten_space) - Flattens a space into the space that contains the [`utils.flatten()`](spaces/utils.md#gymnasium.spaces.utils.flatten) instances
* [`utils.flatten()`](spaces/utils.md#gymnasium.spaces.utils.flatten) - Flattens an instance of a space that is contained within the flattened version of the space
* [`utils.unflatten()`](spaces/utils.md#gymnasium.spaces.utils.unflatten) - The reverse of the [`utils.flatten()`](spaces/utils.md#gymnasium.spaces.utils.flatten) function
* [`utils.is_space_dtype_shape_equiv()`](spaces/utils.md#gymnasium.spaces.utils.is_space_dtype_shape_equiv) - Checks if two spaces have the same dtype and shape, used to check if spaces can be batched together

## Vector utility functions

When vectorizing environments, it is necessary to modify the observation and action spaces for new batched spaces sizes.
Therefore, Gymnasium provides a number of additional functions used when using a space with a Vector environment.

* [`vector.utils.batch_space()`](vector/utils.md#gymnasium.vector.utils.batch_space) - Transforms a space into the equivalent space for `n` users
* [`vector.utils.batch_differing_spaces()`](vector/utils.md#gymnasium.vector.utils.batch_differing_spaces) - Batches a sequence of spaces of the same type that have different parameters, e.g., `Box` spaces with different bounds
* [`vector.utils.concatenate()`](vector/utils.md#gymnasium.vector.utils.concatenate) - Concatenates a space’s samples into a pre-generated space
* [`vector.utils.iterate()`](vector/utils.md#gymnasium.vector.utils.iterate) - Iterate over the batched space’s samples
* [`vector.utils.create_empty_array()`](vector/utils.md#gymnasium.vector.utils.create_empty_array) - Creates an empty sample for an space (generally used with `concatenate`)
* [`vector.utils.create_shared_memory()`](vector/utils.md#gymnasium.vector.utils.create_shared_memory) - Creates a shared memory for asynchronous (multiprocessing) environment
* [`vector.utils.read_from_shared_memory()`](vector/utils.md#gymnasium.vector.utils.read_from_shared_memory) - Reads a shared memory for asynchronous (multiprocessing) environment
* [`vector.utils.write_to_shared_memory()`](vector/utils.md#gymnasium.vector.utils.write_to_shared_memory) - Write to a shared memory for asynchronous (multiprocessing) environment
