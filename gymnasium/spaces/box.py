"""Implementation of a space that represents closed boxes in euclidean space."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, SupportsFloat

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium.spaces.space import Space


def _short_repr(arr: NDArray[Any]) -> str:
    """Create a shortened string representation of a numpy array.

    If arr is a multiple of the all-ones vector, return a string representation of the multiplier.
    Otherwise, return a string representation of the entire array.

    Args:
        arr: The array to represent

    Returns:
        A short representation of the array
    """
    if arr.size != 0 and np.min(arr) == np.max(arr):
        return str(np.min(arr))
    return str(arr)


def is_float_integer(var: Any) -> bool:
    """Checks if a variable is an integer or float."""
    return np.issubdtype(type(var), np.integer) or np.issubdtype(type(var), np.floating)


class Box(Space[NDArray[Any]]):
    r"""A (possibly unbounded) box in :math:`\mathbb{R}^n`.

    Specifically, a Box represents the Cartesian product of n closed intervals.
    Each interval has the form of one of :math:`[a, b]`, :math:`(-\infty, b]`,
    :math:`[a, \infty)`, or :math:`(-\infty, \infty)`.

    There are two common use cases:

    * Identical bound for each dimension::

        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(-1.0, 2.0, (3, 4), float32)

    * Independent bound for each dimension::

        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box([-1. -2.], [2. 4.], (2,), float32)
    """

    def __init__(
        self,
        low: SupportsFloat | NDArray[Any],
        high: SupportsFloat | NDArray[Any],
        shape: Sequence[int] | None = None,
        dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
    ):
        r"""Constructor of :class:`Box`.

        The argument ``low`` specifies the lower bound of each dimension and ``high`` specifies the upper bounds.
        I.e., the space that is constructed will be the product of the intervals :math:`[\text{low}[i], \text{high}[i]]`.

        If ``low`` (or ``high``) is a scalar, the lower bound (or upper bound, respectively) will be assumed to be
        this value across all dimensions.

        Args:
            low (SupportsFloat | np.ndarray): Lower bounds of the intervals. If integer, must be at least ``-2**63``.
            high (SupportsFloat | np.ndarray]): Upper bounds of the intervals. If integer, must be at most ``2**63 - 2``.
            shape (Optional[Sequence[int]]): The shape is inferred from the shape of `low` or `high` `np.ndarray`s with
                `low` and `high` scalars defaulting to a shape of (1,)
            dtype: The dtype of the elements of the space. If this is an integer type, the :class:`Box` is essentially a discrete space.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.

        Raises:
            ValueError: If no shape information is provided (shape is None, low is None and high is None) then a
                value error is raised.
        """
        assert (
            dtype is not None
        ), "Box dtype must be explicitly provided, cannot be None."
        self.dtype = np.dtype(dtype)

        # determine shape if it isn't provided directly
        if shape is not None:
            assert all(
                np.issubdtype(type(dim), np.integer) for dim in shape
            ), f"Expected all shape elements to be an integer, actual type: {tuple(type(dim) for dim in shape)}"
            shape = tuple(int(dim) for dim in shape)  # This changes any np types to int
        elif isinstance(low, np.ndarray):
            shape = low.shape
        elif isinstance(high, np.ndarray):
            shape = high.shape
        elif is_float_integer(low) and is_float_integer(high):
            shape = (1,)
        else:
            raise ValueError(
                f"Box shape is inferred from low and high, expected their types to be np.ndarray, an integer or a float, actual type low: {type(low)}, high: {type(high)}"
            )

        # Capture the boundedness information before replacing np.inf with get_inf
        _low = np.full(shape, low, dtype=float) if is_float_integer(low) else low
        self.bounded_below: NDArray[np.bool_] = -np.inf < _low

        _high = np.full(shape, high, dtype=float) if is_float_integer(high) else high
        self.bounded_above: NDArray[np.bool_] = np.inf > _high

        low = _broadcast(low, self.dtype, shape)
        high = _broadcast(high, self.dtype, shape)

        assert isinstance(low, np.ndarray)
        assert (
            low.shape == shape
        ), f"low.shape doesn't match provided shape, low.shape: {low.shape}, shape: {shape}"
        assert isinstance(high, np.ndarray)
        assert (
            high.shape == shape
        ), f"high.shape doesn't match provided shape, high.shape: {high.shape}, shape: {shape}"

        # check that we don't have invalid low or high
        if np.any(low > high):
            raise ValueError(
                f"Some low values are greater than high, low={low}, high={high}"
            )
        if np.any(np.isposinf(low)):
            raise ValueError(f"No low value can be equal to `np.inf`, low={low}")
        if np.any(np.isneginf(high)):
            raise ValueError(f"No high value can be equal to `-np.inf`, high={high}")

        self._shape: tuple[int, ...] = shape

        low_precision = get_precision(low.dtype)
        high_precision = get_precision(high.dtype)
        dtype_precision = get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:
            gym.logger.warn(f"Box bound precision lowered by casting to {self.dtype}")
        self.low = low.astype(self.dtype)
        self.high = high.astype(self.dtype)

        self.low_repr = _short_repr(self.low)
        self.high_repr = _short_repr(self.high)

        super().__init__(self.shape, self.dtype, seed)

    @property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def is_bounded(self, manner: str = "both") -> bool:
        """Checks whether the box is bounded in some sense.

        Args:
            manner (str): One of ``"both"``, ``"below"``, ``"above"``.

        Returns:
            If the space is bounded

        Raises:
            ValueError: If `manner` is neither ``"both"`` nor ``"below"`` or ``"above"``
        """
        below = bool(np.all(self.bounded_below))
        above = bool(np.all(self.bounded_above))
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError(
                f"manner is not in {{'below', 'above', 'both'}}, actual value: {manner}"
            )

    def sample(self, mask: None = None) -> NDArray[Any]:
        r"""Generates a single random sample inside the Box.

        In creating a sample of the box, each coordinate is sampled (independently) from a distribution
        that is chosen according to the form of the interval:

        * :math:`[a, b]` : uniform distribution
        * :math:`[a, \infty)` : shifted exponential distribution
        * :math:`(-\infty, b]` : shifted negative exponential distribution
        * :math:`(-\infty, \infty)` : normal distribution

        Args:
            mask: A mask for sampling values from the Box space, currently unsupported.

        Returns:
            A sampled value from the Box
        """
        if mask is not None:
            raise gym.error.Error(
                f"Box.sample cannot be provided a mask, actual value: {mask}"
            )

        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = self.np_random.normal(size=unbounded[unbounded].shape)

        sample[low_bounded] = (
            self.np_random.exponential(size=low_bounded[low_bounded].shape)
            + self.low[low_bounded]
        )

        sample[upp_bounded] = (
            -self.np_random.exponential(size=upp_bounded[upp_bounded].shape)
            + high[upp_bounded]
        )

        sample[bounded] = self.np_random.uniform(
            low=self.low[bounded], high=high[bounded], size=bounded[bounded].shape
        )

        if self.dtype.kind in ["i", "u", "b"]:
            sample = np.floor(sample)

        return sample.astype(self.dtype)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, np.ndarray):
            gym.logger.warn("Casting input x to numpy array.")
            try:
                x = np.asarray(x, dtype=self.dtype)
            except (ValueError, TypeError):
                return False

        return bool(
            np.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def to_jsonable(self, sample_n: Sequence[NDArray[Any]]) -> list[list]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n: Sequence[float | int]) -> list[NDArray[Any]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [np.asarray(sample, dtype=self.dtype) for sample in sample_n]

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include bounds, shape and dtype.
        If a bound is uniform, only the corresponding scalar will be given to avoid redundant and ugly strings.

        Returns:
            A representation of the space
        """
        return f"Box({self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype})"

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        return (
            isinstance(other, Box)
            and (self.shape == other.shape)
            # and (self.dtype == other.dtype)
            and np.allclose(self.low, other.low)
            and np.allclose(self.high, other.high)
        )

    def __setstate__(self, state: Iterable[tuple[str, Any]] | Mapping[str, Any]):
        """Sets the state of the box for unpickling a box with legacy support."""
        super().__setstate__(state)

        # legacy support through re-adding "low_repr" and "high_repr" if missing from pickled state
        if not hasattr(self, "low_repr"):
            self.low_repr = _short_repr(self.low)

        if not hasattr(self, "high_repr"):
            self.high_repr = _short_repr(self.high)


def get_precision(dtype: np.dtype) -> SupportsFloat:
    """Get precision of a data type."""
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).precision
    else:
        return np.inf


def _broadcast(
    value: SupportsFloat | NDArray[Any],
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> NDArray[Any]:
    """Handle infinite bounds and broadcast at the same time if needed.

    This is needed primarily because:
        >>> import numpy as np
        >>> np.full((2,), np.inf, dtype=np.int32)
        array([-2147483648, -2147483648], dtype=int32)
    """
    if is_float_integer(value):
        if np.isneginf(value) and np.dtype(dtype).kind == "i":
            value = np.iinfo(dtype).min + 2
        elif np.isposinf(value) and np.dtype(dtype).kind == "i":
            value = np.iinfo(dtype).max - 2

        return np.full(shape, value, dtype=dtype)

    elif isinstance(value, np.ndarray):
        # this is needed because we can't stuff np.iinfo(int).min into an array of dtype float
        casted_value = value.astype(dtype)

        # change bounds only if values are negative or positive infinite
        if np.dtype(dtype).kind == "i":
            casted_value[np.isneginf(value)] = np.iinfo(dtype).min + 2
            casted_value[np.isposinf(value)] = np.iinfo(dtype).max - 2

        return casted_value

    else:
        # only np.ndarray allowed beyond this point
        raise TypeError(
            f"Unknown dtype for `value`, expected `np.ndarray` or float/integer, got {type(value)}"
        )
