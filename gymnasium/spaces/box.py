"""Implementation of a space that represents closed boxes in euclidean space."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast, overload

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium.spaces.space import Space

if TYPE_CHECKING:
    from typing_extensions import TypeIs, TypeVar

    _ScalarT_co = TypeVar(
        "_ScalarT_co",
        bound="_RealScalar",
        default="_RealScalar",
        covariant=True,
    )
else:
    from typing import TypeVar

    _ScalarT_co = TypeVar("_ScalarT_co", bound="_RealScalar", covariant=True)

_RealScalar: TypeAlias = np.floating | np.integer
_RealArrayLike: TypeAlias = int | float | _RealScalar | NDArray[_RealScalar]
_ToSeed: TypeAlias = int | np.random.Generator
_ToShape: TypeAlias = Sequence[int | np.integer]

_ScalarT = TypeVar("_ScalarT", bound=_RealScalar)
_FloatScalarT = TypeVar("_FloatScalarT", bound=np.floating)
_IntScalarT = TypeVar("_IntScalarT", bound=np.integer)

_ToDType: TypeAlias = type[_ScalarT] | np.dtype[_ScalarT]


def array_short_repr(arr: NDArray[Any]) -> str:
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


def is_float_integer(var: object) -> TypeIs[float | np.floating | np.integer]:
    """Checks if a scalar variable is an integer or float (does not include bool)."""
    return isinstance(var, (int, float, np.integer, np.floating)) and not isinstance(
        var, bool
    )


class Box(Space[NDArray[_ScalarT_co]]):
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

    dtype: np.dtype[_ScalarT_co]
    _shape: tuple[int, ...]

    low: NDArray[_ScalarT_co]
    high: NDArray[_ScalarT_co]

    bounded_below: NDArray[np.bool]
    bounded_above: NDArray[np.bool]

    low_repr: str | None
    high_repr: str | None

    @overload
    def __init__(
        self: Box[np.float32],
        low: _RealArrayLike,
        high: _RealArrayLike,
        shape: _ToShape | None = None,
        dtype: _ToDType[np.float32] = np.float32,
        seed: _ToSeed | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: Box[_ScalarT],
        low: _RealArrayLike,
        high: _RealArrayLike,
        shape: _ToShape | None = None,
        *,
        dtype: _ToDType[_ScalarT],
        seed: _ToSeed | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: Box[_ScalarT],
        low: _RealArrayLike,
        high: _RealArrayLike,
        shape: _ToShape | None,
        dtype: _ToDType[_ScalarT],
        seed: _ToSeed | None = None,
    ) -> None: ...
    def __init__(
        self,
        low: _RealArrayLike,
        high: _RealArrayLike,
        shape: _ToShape | None = None,
        dtype: _ToDType[_ScalarT_co | np.float32] = np.float32,
        seed: _ToSeed | None = None,
    ) -> None:
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
        # determine dtype
        if dtype is None:
            raise ValueError("Box dtype must be explicitly provided, cannot be None.")
        self.dtype = np.dtype(dtype)

        #  * check that dtype is an accepted dtype
        if self.dtype.kind not in ("i", "u", "f", "b"):
            raise ValueError(
                f"Invalid Box dtype ({self.dtype}), must be an integer, floating, or bool dtype"
            )

        # determine shape
        if shape is not None:
            if not isinstance(shape, Iterable):
                raise TypeError(
                    f"Expected Box shape to be an iterable, actual type={type(shape)}"
                )
            elif not all(isinstance(dim, (int, np.integer)) for dim in shape):
                raise TypeError(
                    f"Expected all Box shape elements to be integer, actual type={tuple(type(dim) for dim in shape)}"
                )

            # Casts the `shape` argument to tuple[int, ...] (otherwise dim can `np.int64`)
            shape = tuple(int(dim) for dim in shape)
        elif isinstance(low, np.ndarray) and isinstance(high, np.ndarray):
            if low.shape != high.shape:
                raise ValueError(
                    f"Box low.shape and high.shape don't match, low.shape={low.shape}, high.shape={high.shape}"
                )
            shape = low.shape
        elif isinstance(low, np.ndarray):
            shape = low.shape
        elif isinstance(high, np.ndarray):
            shape = high.shape
        elif is_float_integer(low) and is_float_integer(high):
            shape = (1,)  # low and high are scalars
        else:
            raise ValueError(
                "Box shape is not specified, therefore inferred from low and high. Expected low and high to be np.ndarray, integer, or float."
                f"Actual types low={type(low)}, high={type(high)}"
            )
        self._shape = shape

        # Cast scalar values to `np.ndarray` and capture the boundedness information
        # disallowed cases
        #  * out of range - this must be done before casting to low and high otherwise, the value is within dtype and cannot be out of range
        #  * nan - must be done beforehand as int dtype can cast `nan` to another value
        #  * unsign int inf and -inf - special case that is disallowed

        if self.dtype.kind == "b":
            dtype_min, dtype_max = 0, 1
        elif self.dtype.kind == "f":
            finfo = np.finfo(cast(np.dtype[np.floating], self.dtype))
            dtype_min, dtype_max = float(finfo.min), float(finfo.max)
        else:
            iinfo = np.iinfo(cast(np.dtype[np.integer], self.dtype))
            dtype_min, dtype_max = int(iinfo.min), int(iinfo.max)

        # Cast `low` and `high` to ndarray for the dtype min and max for out of range tests
        self.low, self.bounded_below = self._cast_low(low, dtype_min)
        self.high, self.bounded_above = self._cast_high(high, dtype_max)

        # recheck shape for case where shape and (low or high) are provided
        if self.low.shape != shape:
            raise ValueError(
                f"Box low.shape doesn't match provided shape, low.shape={self.low.shape}, shape={self.shape}"
            )
        if self.high.shape != shape:
            raise ValueError(
                f"Box high.shape doesn't match provided shape, high.shape={self.high.shape}, shape={self.shape}"
            )

        # check that low <= high
        if np.any(self.low > self.high):
            raise ValueError(
                f"Box all low values must be less than or equal to high (some values break this), low={self.low}, high={self.high}"
            )

        self.low_repr = None
        self.high_repr = None

        super().__init__(self.shape, self.dtype, seed)

    def _cast_low(
        self, low: _RealArrayLike, dtype_min: float
    ) -> tuple[NDArray[_ScalarT_co], NDArray[np.bool]]:
        """Casts the input Box low value to ndarray with provided dtype.

        Args:
            low: The input box low value
            dtype_min: The dtype's minimum value

        Returns:
            The updated low value and for what values the input is bounded (below)
        """
        if is_float_integer(low):
            bounded_below = np.full(self.shape, -np.inf < low)

            if np.isnan(low):
                raise ValueError(f"No low value can be equal to `np.nan`, low={low}")
            elif np.isneginf(low):
                if self.dtype.kind == "i":  # signed int
                    low = dtype_min
                elif self.dtype.kind in {"u", "b"}:  # unsigned int and bool
                    raise ValueError(
                        f"Box unsigned int dtype don't support `-np.inf`, low={low}"
                    )
            elif low < dtype_min:
                raise ValueError(
                    f"Box low is out of bounds of the dtype range, low={low}, min dtype={dtype_min}"
                )

            low = np.full(self.shape, low, dtype=self.dtype)
            return low, bounded_below
        else:  # cast for low - array
            if not isinstance(low, np.ndarray):
                raise ValueError(
                    f"Box low must be a np.ndarray, integer, or float, actual type={type(low)}"
                )
            elif low.dtype.kind not in ("f", "i", "u", "b"):
                raise ValueError(
                    f"Box low must be a floating, integer, or bool dtype, actual dtype={low.dtype}"
                )
            elif np.any(np.isnan(low)):
                raise ValueError(f"No low value can be equal to `np.nan`, low={low}")

            bounded_below = -np.inf < low

            neginf = np.isneginf(low)
            if np.any(neginf):
                if self.dtype.kind == "i":  # signed int
                    low[neginf] = dtype_min
                elif self.dtype.kind in {"u", "b"}:  # unsigned int and bool
                    raise ValueError(
                        f"Box unsigned int dtype don't support `-np.inf`, low={low}"
                    )
            elif low.dtype != self.dtype and np.any(low < dtype_min):
                raise ValueError(
                    f"Box low is out of bounds of the dtype range, low={low}, min dtype={dtype_min}"
                )

            if low.dtype.kind == "f" and self.dtype.kind == "f":
                dtype_self = cast("np.dtype[np.floating]", self.dtype)
                dtype_low = cast("np.dtype[np.floating]", low.dtype)
                if np.finfo(dtype_self).precision < np.finfo(dtype_low).precision:
                    gym.logger.warn(
                        f"Box low's precision lowered by casting to {self.dtype}, current low.dtype={low.dtype}"
                    )
            return low.astype(self.dtype), bounded_below

    def _cast_high(
        self, high: _RealArrayLike, dtype_max: float
    ) -> tuple[NDArray[_ScalarT_co], NDArray[np.bool]]:
        """Casts the input Box high value to ndarray with provided dtype.

        Args:
            high: The input box high value
            dtype_max: The dtype's maximum value

        Returns:
            The updated high value and for what values the input is bounded (above)
        """
        if is_float_integer(high):
            bounded_above = np.full(self.shape, high < np.inf)

            if np.isnan(high):
                raise ValueError(f"No high value can be equal to `np.nan`, high={high}")
            elif np.isposinf(high):
                if self.dtype.kind == "i":  # signed int
                    high = dtype_max
                elif self.dtype.kind in {"u", "b"}:  # unsigned int
                    raise ValueError(
                        f"Box unsigned int dtype don't support `np.inf`, high={high}"
                    )
            elif high > dtype_max:
                raise ValueError(
                    f"Box high is out of bounds of the dtype range, high={high}, max dtype={dtype_max}"
                )

            high = np.full(self.shape, high, dtype=self.dtype)
            return high, bounded_above
        else:
            if not isinstance(high, np.ndarray):
                raise ValueError(
                    f"Box high must be a np.ndarray, integer, or float, actual type={type(high)}"
                )
            elif high.dtype.kind not in ("f", "i", "u", "b"):
                raise ValueError(
                    f"Box high must be a floating or integer dtype, actual dtype={high.dtype}"
                )
            elif np.any(np.isnan(high)):
                raise ValueError(f"No high value can be equal to `np.nan`, high={high}")

            bounded_above = high < np.inf

            posinf = np.isposinf(high)
            if np.any(posinf):
                if self.dtype.kind == "i":  # signed int
                    high[posinf] = dtype_max
                elif self.dtype.kind in {"u", "b"}:  # unsigned int
                    raise ValueError(
                        f"Box unsigned int dtype don't support `np.inf`, high={high}"
                    )
            elif high.dtype != self.dtype and np.any(dtype_max < high):
                raise ValueError(
                    f"Box high is out of bounds of the dtype range, high={high}, max dtype={dtype_max}"
                )

            if high.dtype.kind == "f" and self.dtype.kind == "f":
                dtype_self = cast("np.dtype[np.floating]", self.dtype)
                dtype_high = cast("np.dtype[np.floating]", high.dtype)
                if np.finfo(dtype_self).precision < np.finfo(dtype_high).precision:
                    gym.logger.warn(
                        f"Box high's precision lowered by casting to {self.dtype}, current high.dtype={high.dtype}"
                    )
            return high.astype(self.dtype), bounded_above

    @property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape

    @property
    def is_np_flattenable(self) -> Literal[True]:
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def is_bounded(self, manner: Literal["both", "below", "above"] = "both") -> bool:
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

    def sample(
        self, mask: None = None, probability: None = None
    ) -> NDArray[_ScalarT_co]:
        r"""Generates a single random sample inside the Box.

        In creating a sample of the box, each coordinate is sampled (independently) from a distribution
        that is chosen according to the form of the interval:

        * :math:`[a, b]` : uniform distribution
        * :math:`[a, \infty)` : shifted exponential distribution
        * :math:`(-\infty, b]` : shifted negative exponential distribution
        * :math:`(-\infty, \infty)` : normal distribution

        Args:
            mask: A mask for sampling values from the Box space, currently unsupported.
            probability: A probability mask for sampling values from the Box space, currently unsupported.

        Returns:
            A sampled value from the Box
        """
        if mask is not None:
            raise gym.error.Error(
                f"Box.sample cannot be provided a mask, actual value: {mask}"
            )
        elif probability is not None:
            raise gym.error.Error(
                f"Box.sample cannot be provided a probability mask, actual value: {probability}"
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

        # clip values that would underflow/overflow
        if np.issubdtype(self.dtype, np.integer):
            iinfo = np.iinfo(cast("np.dtype[np.integer]", self.dtype))
            dtype_min, dtype_max = iinfo.min, iinfo.max
            if np.issubdtype(self.dtype, np.signedinteger):
                dtype_min += 2
                dtype_max -= 2
            sample = sample.clip(min=dtype_min, max=dtype_max)

        sample = sample.astype(self.dtype)

        # float64 values have lower than integer precision near int64 min/max, so clip
        # again in case something has been cast to an out-of-bounds value
        if self.dtype == np.int64:
            sample = sample.clip(min=self.low, max=self.high)

        return sample

    def contains(self, x: np.ndarray | Any) -> bool:
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

    def to_jsonable(self, sample_n: Iterable[np.ndarray]) -> list[list]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(
        self, sample_n: Iterable[float | list]
    ) -> list[NDArray[_ScalarT_co]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [np.asarray(sample, dtype=self.dtype) for sample in sample_n]

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include bounds, shape and dtype.
        If a bound is uniform, only the corresponding scalar will be given to avoid redundant and ugly strings.

        Returns:
            A representation of the space
        """
        if self.low_repr is None:
            self.low_repr = array_short_repr(self.low)
        if self.high_repr is None:
            self.high_repr = array_short_repr(self.high)
        return f"Box({self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype})"

    def __eq__(self, other: object, /) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        return (
            isinstance(other, Box)
            and (self.shape == other.shape)
            and (self.dtype == other.dtype)
            and np.allclose(self.low, other.low)
            and np.allclose(self.high, other.high)
        )

    def __setstate__(
        self, state: Iterable[tuple[str, Any]] | Mapping[str, Any]
    ) -> None:
        """Sets the state of the box for unpickling a box with legacy support."""
        super().__setstate__(state)

        # legacy support through re-adding "low_repr" and "high_repr" if missing from pickled state
        if not hasattr(self, "low_repr"):
            self.low_repr = array_short_repr(self.low)

        if not hasattr(self, "high_repr"):
            self.high_repr = array_short_repr(self.high)
