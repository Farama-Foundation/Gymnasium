# Fundamental Spaces

### *class* gymnasium.spaces.Box(low: int | float | floating | integer | NDArray[floating | integer], high: int | float | floating | integer | NDArray[floating | integer], shape: Sequence[int | integer] | None = None, dtype: type[float32] | dtype[float32] = np.float32, seed: int | Generator | None = None)

### *class* gymnasium.spaces.Box(low: int | float | floating | integer | NDArray[floating | integer], high: int | float | floating | integer | NDArray[floating | integer], shape: Sequence[int | integer] | None = None, \*, dtype: type[\_ScalarT] | dtype[\_ScalarT], seed: int | Generator | None = None)

### *class* gymnasium.spaces.Box(low: int | float | floating | integer | NDArray[floating | integer], high: int | float | floating | integer | NDArray[floating | integer], shape: Sequence[int | integer] | None, dtype: type[\_ScalarT] | dtype[\_ScalarT], seed: int | Generator | None = None)

A (possibly unbounded) box in $\mathbb{R}^n$.

Specifically, a Box represents the Cartesian product of n closed intervals.
Each interval has the form of one of $[a, b]$, $(-\infty, b]$,
$[a, \infty)$, or $(-\infty, \infty)$.

There are two common use cases:

* Identical bound for each dimension:
  ```default
  >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
  Box(-1.0, 2.0, (3, 4), float32)
  ```
* Independent bound for each dimension:
  ```default
  >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
  Box([-1. -2.], [2. 4.], (2,), float32)
  ```

* **Parameters:**
  * **low** (*SupportsFloat* *|* *np.ndarray*) – Lower bounds of the intervals. If integer, must be at least `-2**63`.
  * **high** (*SupportsFloat* *|* *np.ndarray* *]*) – Upper bounds of the intervals. If integer, must be at most `2**63 - 2`.
  * **shape** (*Optional* *[*[*Sequence*](composite.md#gymnasium.spaces.Sequence) *[**int* *]* *]*) – The shape is inferred from the shape of low or high np.ndarray\`s with
    \`low and high scalars defaulting to a shape of (1,)
  * **dtype** – The dtype of the elements of the space. If this is an integer type, the [`Box`](#gymnasium.spaces.Box) is essentially a discrete space.
  * **seed** – Optionally, you can use this argument to seed the RNG that is used to sample from the space.
* **Raises:**
  **ValueError** – If no shape information is provided (shape is None, low is None and high is None) then a
      value error is raised.

#### sample(mask: None = None, probability: None = None) → NDArray[\_ScalarT_co]

Generates a single random sample inside the Box.

In creating a sample of the box, each coordinate is sampled (independently) from a distribution
that is chosen according to the form of the interval:

* $[a, b]$ : uniform distribution
* $[a, \infty)$ : shifted exponential distribution
* $(-\infty, b]$ : shifted negative exponential distribution
* $(-\infty, \infty)$ : normal distribution

* **Parameters:**
  * **mask** – A mask for sampling values from the Box space, currently unsupported.
  * **probability** – A probability mask for sampling values from the Box space, currently unsupported.
* **Returns:**
  **A sampled value from the Box**

#### seed(seed: int | None = None) → int | Any

Seed the pseudorandom number generator (PRNG) of this space and, if applicable, the PRNGs of subspaces.

* **Parameters:**
  **seed** – The seed value for the space. This is expanded for composite spaces to accept multiple values. For further details, please refer to the space’s documentation.
* **Returns:**
  **The seed values used for all the PRNGs, for composite spaces this can be a tuple or dictionary of values.**

#### is_bounded(manner: Literal['both', 'below', 'above'] = 'both') → bool

Checks whether the box is bounded in some sense.

* **Parameters:**
  **manner** (*str*) – One of `"both"`, `"below"`, `"above"`.
* **Returns:**
  **If the space is bounded**
* **Raises:**
  **ValueError** – If manner is neither `"both"` nor `"below"` or `"above"`

### *class* gymnasium.spaces.Discrete(n: int | integer[Any], seed: int | Generator | None = None, start: int | integer[Any] = 0, \*, dtype: type[\_IntegerT_co] | dtype[\_IntegerT_co])

### *class* gymnasium.spaces.Discrete(n: int | integer[Any], seed: int | Generator | None = None, start: int | integer[Any] = 0, \*, dtype: str)

### *class* gymnasium.spaces.Discrete(n: int | integer[Any], seed: int | Generator | None = None, start: int | integer[Any] = 0, dtype: type[int64] = np.int64)

A space consisting of finitely many elements.

This class represents a finite subset of integers, more specifically a set of the form $\{ a, a+1, \dots, a+n-1 \}$.

### Example

```pycon
>>> from gymnasium.spaces import Discrete
>>> observation_space = Discrete(2, seed=42) # {0, 1}
>>> observation_space.sample()
np.int64(0)
>>> observation_space = Discrete(3, start=-1, seed=42)  # {-1, 0, 1}
>>> observation_space.sample()
np.int64(-1)
>>> observation_space.sample(mask=np.array([0,0,1], dtype=np.int8))
np.int64(1)
>>> observation_space.sample(probability=np.array([0,0,1], dtype=np.float64))
np.int64(1)
>>> observation_space.sample(probability=np.array([0,0.3,0.7], dtype=np.float64))
np.int64(1)
>>> observation_space = Discrete(3, seed=37, dtype=np.int32) # specify dtype for samples
>>> observation_space.sample()
np.int32(0)
```

* **Parameters:**
  * **n** (*int*) – The number of elements of this space.
  * **seed** – Optionally, you can use this argument to seed the RNG that is used to sample from the `Dict` space.
  * **start** (*int*) – The smallest element of this space.
  * **dtype** – The space type, for example, `int`, `np.int64`, `np.int32`, or `np.uint8`.

#### sample(mask: NDArray[int8] | None = None, probability: NDArray[int8] | None = None) → \_IntegerT_co

Generates a single random sample from this space.

A sample will be chosen uniformly at random with the mask if provided, or it will be chosen according to a specified probability distribution if the probability mask is provided.

* **Parameters:**
  * **mask** – An optional mask for if an action can be selected.
    Expected np.ndarray of shape `(n,)` and dtype `np.int8` where `1` represents valid actions and `0` invalid / infeasible actions.
    If there are no possible actions (i.e. `np.all(mask == 0)`) then `space.start` will be returned.
  * **probability** – An optional probability mask describing the probability of each action being selected.
    Expected np.ndarray of shape `(n,)` and dtype `np.float64` where each value is in the range `[0, 1]` and the sum of all values is 1.
    If the values do not sum to 1, an exception will be thrown.
* **Returns:**
  **A sampled integer from the space**

#### seed(seed: int | None = None) → int | Any

Seed the pseudorandom number generator (PRNG) of this space and, if applicable, the PRNGs of subspaces.

* **Parameters:**
  **seed** – The seed value for the space. This is expanded for composite spaces to accept multiple values. For further details, please refer to the space’s documentation.
* **Returns:**
  **The seed values used for all the PRNGs, for composite spaces this can be a tuple or dictionary of values.**

### *class* gymnasium.spaces.MultiBinary(n: NDArray[integer[Any]] | Sequence[int] | int | integer[Any], seed: int | Generator | None = None)

An n-shape binary space.

Elements of this space are binary arrays of a shape that is fixed during construction.

### Example

```pycon
>>> from gymnasium.spaces import MultiBinary
>>> observation_space = MultiBinary(5, seed=42)
>>> observation_space.sample()
array([1, 0, 1, 0, 1], dtype=int8)
>>> observation_space = MultiBinary([3, 2], seed=42)
>>> observation_space.sample()
array([[1, 0],
       [1, 0],
       [1, 1]], dtype=int8)
```

* **Parameters:**
  * **n** – This will fix the shape of elements of the space. It can either be an integer (if the space is flat)
    or some sort of sequence (tuple, list or np.ndarray) if there are multiple axes.
  * **seed** – Optionally, you can use this argument to seed the RNG that is used to sample from the space.

#### sample(mask: NDArray[int8] | None = None, probability: NDArray[int8] | None = None) → NDArray[int8]

Generates a single random sample from this space.

A sample is drawn by independent, fair coin tosses (one toss per binary variable of the space).

* **Parameters:**
  * **mask** – An optional `np.ndarray` to mask samples with expected shape of `space.shape`.
    For `mask == 0` then the samples will be `0`, for a `mask == 1` then the samples will be `1`.
    For random samples, using a mask value of `2`.
    The expected mask shape is the space shape and mask dtype is `np.int8`.
  * **probability** – An optional `np.ndarray` to mask samples with expected shape of space.shape where each element
    represents the probability of the corresponding sample element being a 1.
    The expected mask shape is the space shape and mask dtype is `np.float64`.
* **Returns:**
  **Sampled values from space**

#### seed(seed: int | None = None) → int | Any

Seed the pseudorandom number generator (PRNG) of this space and, if applicable, the PRNGs of subspaces.

* **Parameters:**
  **seed** – The seed value for the space. This is expanded for composite spaces to accept multiple values. For further details, please refer to the space’s documentation.
* **Returns:**
  **The seed values used for all the PRNGs, for composite spaces this can be a tuple or dictionary of values.**

### *class* gymnasium.spaces.MultiDiscrete(nvec: NDArray[integer[Any]] | list[int], dtype: type[int64] = np.int64, seed: int | Generator | None = None, start: NDArray[integer[Any]] | list[int] | None = None)

### *class* gymnasium.spaces.MultiDiscrete(nvec: NDArray[integer[Any]] | list[int], dtype: type[\_IntegerT_co] | dtype[\_IntegerT_co], seed: int | Generator | None = None, start: NDArray[integer[Any]] | list[int] | None = None)

### *class* gymnasium.spaces.MultiDiscrete(nvec: NDArray[integer[Any]] | list[int], dtype: str, seed: int | Generator | None = None, start: NDArray[integer[Any]] | list[int] | None = None)

This represents the cartesian product of arbitrary [`Discrete`](#gymnasium.spaces.Discrete) spaces.

It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space.

#### NOTE
Some environment wrappers assume a value of 0 always represents the NOOP action.

e.g. Nintendo Game Controller - Can be conceptualized as 3 discrete action spaces:

1. Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
2. Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
3. Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

It can be initialized as `MultiDiscrete([ 5, 2, 2 ])` such that a sample might be `array([3, 1, 0])`.

Although this feature is rarely used, [`MultiDiscrete`](#gymnasium.spaces.MultiDiscrete) spaces may also have several axes
if `nvec` has several axes:

### Example

```pycon
>>> from gymnasium.spaces import MultiDiscrete
>>> import numpy as np
>>> observation_space = MultiDiscrete(np.array([[1, 2], [3, 4]]), seed=42)
>>> observation_space.sample()
array([[0, 0],
       [2, 2]])
```

* **Parameters:**
  * **nvec** – vector of counts of each categorical variable. This will usually be a list of integers. However,
    you may also pass a more complicated numpy array if you’d like the space to have several axes.
  * **dtype** – This should be some kind of integer type.
  * **seed** – Optionally, you can use this argument to seed the RNG that is used to sample from the space.
  * **start** – Optionally, the starting value the element of each class will take (defaults to 0).

#### sample(mask: tuple[NDArray[int8], ...] | None = None, probability: tuple[NDArray[int8], ...] | None = None) → NDArray[\_IntegerT_co]

Generates a single random sample from this space.

* **Parameters:**
  * **mask** – An optional mask for multi-discrete, expects tuples with a `np.ndarray` mask in the position of each
    action with shape `(n,)` where `n` is the number of actions and `dtype=np.int8`.
    Only `mask values == 1` are possible to sample unless all mask values for an action are `0` then the default action `self.start` (the smallest element) is sampled.
  * **probability** – An optional probability mask for multi-discrete, expects tuples with a `np.ndarray` probability mask in the position of each
    action with shape `(n,)` where `n` is the number of actions and `dtype=np.float64`.
    Only probability mask values within `[0,1]` are possible to sample as long as the sum of all values is `1`.
* **Returns:**
  **An \`\`np.ndarray\`\` of :meth:\`Space.shape\`**

#### seed(seed: int | None = None) → int | Any

Seed the pseudorandom number generator (PRNG) of this space and, if applicable, the PRNGs of subspaces.

* **Parameters:**
  **seed** – The seed value for the space. This is expanded for composite spaces to accept multiple values. For further details, please refer to the space’s documentation.
* **Returns:**
  **The seed values used for all the PRNGs, for composite spaces this can be a tuple or dictionary of values.**

### *class* gymnasium.spaces.Text(max_length: int, \*, min_length: int = 1, charset: frozenset[str] | str = alphanumeric, seed: int | Generator | None = None)

A space representing a string comprised of characters from a given charset.

### Example

```pycon
>>> from gymnasium.spaces import Text
>>> # {"a", "B5", "hello", ...}
>>> Text(5)
Text(1, 5, charset=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz)
>>> # {"0", "42", "0123456789", ...}
>>> import string
>>> Text(min_length = 1,
...      max_length = 10,
...      charset = string.digits)
Text(1, 10, charset=0123456789)
```

* **Parameters:**
  * **min_length** (*int*) – Minimum text length (in characters). Defaults to 1 to prevent empty strings.
  * **max_length** (*int*) – Maximum text length (in characters).
  * **charset** (*Union* *[**set* *]* *,* *str*) – Character set, defaults to the lower and upper english alphabet plus latin digits.
    Each element must be a single character.
  * **seed** – The seed for sampling from the space.
* **Raises:**
  **ValueError** – If any element of `charset` is not a single character.

#### sample(mask: None | tuple[int | None, NDArray[int8] | None] = None, probability: None | tuple[int | None, NDArray[float64] | None] = None) → str

Generates a single random sample from this space with by default a random length between `min_length` and `max_length` and sampled from the `charset`.

* **Parameters:**
  * **mask** – An optional tuples of length and mask for the text.
    The length is expected to be between the `min_length` and `max_length`.
    Otherwise, a random integer between `min_length` and `max_length` is selected.
    For the mask, we expect a numpy array of length of the charset passed with `dtype == np.int8`.
    If the character mask is all zero, an empty string is returned only when `min_length == 0`,
    even if a positive length was requested. Otherwise, a `ValueError` is raised.
  * **probability** – An optional tuples of length and probability mask for the text.
    The length is expected to be between the `min_length` and `max_length`.
    Otherwise, a random integer between `min_length` and `max_length` is selected.
    For the probability mask, we expect a numpy array of length of the charset passed with `dtype == np.float64`.
    The sum of the probability mask should be 1, otherwise an exception is raised.
* **Returns:**
  **A sampled string from the space**

### Example

```pycon
>>> from gymnasium.spaces import Text
>>> import numpy as np
>>> mask = np.zeros(2, dtype=np.int8)
>>> Text(3, min_length=0, charset="ab").sample(mask=(3, mask))
''
>>> Text(3, charset="ab").sample(mask=(3, mask))
Traceback (most recent call last):
    ...
ValueError: Trying to sample with a minimum length > 0 (actual minimum length=1) but the character mask is all zero meaning that no character could be sampled.
```

#### seed(seed: int | None = None) → int | Any

Seed the pseudorandom number generator (PRNG) of this space and, if applicable, the PRNGs of subspaces.

* **Parameters:**
  **seed** – The seed value for the space. This is expanded for composite spaces to accept multiple values. For further details, please refer to the space’s documentation.
* **Returns:**
  **The seed values used for all the PRNGs, for composite spaces this can be a tuple or dictionary of values.**
