"""Implementation of a space that represents closed boxes in euclidean space."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import ModuleType
from typing import Any, SupportsFloat

import array_api_extra as xpx
import numpy as np
from array_api_compat import array_namespace, device, is_array_api_obj
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium.spaces.space import Space


# See also https://github.com/data-apis/array-api-typing/
Array = Any  # TODO: Switch to ArrayAPI type once https://github.com/data-apis/array-api/pull/589 is merged
Device = Any  # TODO: Switch to ArrayAPI type if available
DType = Any  # TODO: Switch to ArrayAPI DTypes


def array_short_repr(arr: Array) -> str:
    """Create a shortened string representation of a numpy array.

    If arr is a multiple of the all-ones vector, return a string representation of the multiplier.
    Otherwise, return a string representation of the entire array.

    Args:
        arr: The array to represent

    Returns:
        A short representation of the array
    """
    xp = array_namespace(arr)
    if arr.size != 0 and xp.min(arr) == xp.max(arr):
        return str(xp.min(arr))
    return str(arr)


def is_float_integer(var: Any) -> bool:
    """Checks if a scalar variable is an integer or float (does not include bool)."""
    return np.issubdtype(type(var), np.integer) or np.issubdtype(type(var), np.floating)


class Box(Space[NDArray[Any]]):
    r"""A (possibly unbounded) box in :math:`\mathbb{R}^n`.

    Specifically, a Box represents the Cartesian product of n closed intervals.
    Each interval has the form of one of :math:`[a, b]`, :math:`(-\infty, b]`,
    :math:`[a, \infty)`, or :math:`(-\infty, \infty)`.

    There are two common use cases:

    * Identical bound for each dimension::

        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(-1.0, 2.0, (3, 4), float32, cpu)

    * Independent bound for each dimension::

        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box([-1. -2.], [2. 4.], (2,), float32, cpu)
    """

    _dtype_kinds = ("real floating", "integral", "bool")

    def __init__(
        self,
        low: SupportsFloat | Array,
        high: SupportsFloat | Array,
        shape: Sequence[int] | None = None,
        dtype: DType | str = "float32",
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
        # determine the Array API framework we are dealing with
        try:
            xp = array_namespace(low, high)
        except TypeError:
            xp = array_namespace(np.empty(0))
        self.dtype = self._determine_dtype(dtype, xp)
        self._check_low_high(low, high, xp)
        self.device = self._determine_device(low, high, xp)
        self._shape: tuple[int, ...] = self._determine_shape(low, high, shape, xp)

        # Cast scalar values to `xp.Array` and capture the boundedness information
        self.low, self.bounded_below, _ = self._to_array(low, xp)
        self.high, _, self.bounded_above = self._to_array(high, xp)

        if xp.any(self.low > self.high):  # Make sure that low <= high
            raise ValueError(
                f"Box all low values must be less than or equal to high (some values break this), low={self.low}, high={self.high}"
            )

        super().__init__(self.shape, self.dtype, seed)

    def _to_array(
        self, x: SupportsFloat | Array, xp: ModuleType
    ) -> tuple[Array, np.ndarray, np.ndarray]:
        """Cast the input x to an Array with provided dtype and bounds."""
        # Disallowed cases:
        #  * nan - must be done beforehand as int dtype can cast `nan` to another value
        #  * out of range - this must be done before casting to low and high otherwise, the value is
        #    within dtype and cannot be out of range
        #  * unsign int inf and -inf - special case that is disallowed
        is_array = is_array_api_obj(x)
        if xp.any(xp.isnan(x)):
            raise ValueError(f"No value can be equal to `np.nan`, x={x}")
        x = xp.asarray(x)

        # Check for out of range values
        if xp.isdtype(self.dtype, "integral"):
            dtype_min, dtype_max = xp.iinfo(self.dtype).min, xp.iinfo(self.dtype).max
        elif xp.isdtype(self.dtype, "real floating"):
            dtype_min, dtype_max = xp.finfo(self.dtype).min, xp.finfo(self.dtype).max
        elif xp.isdtype(self.dtype, "bool"):
            dtype_min, dtype_max = 0, 1
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if xp.any((x < dtype_min)[~xp.isinf(x)]):
            raise ValueError(
                f"Box is out of bounds of the dtype range, {x}, max dtype={dtype_max}"
            )
        if xp.any((x > dtype_max)[~xp.isinf(x)]):
            raise ValueError(
                f"Box is out of bounds of the dtype range, {x}, min dtype={dtype_min}"
            )

        # Check for inf values in bool or unsigned int
        if xp.isdtype(self.dtype, "unsigned integer") or xp.isdtype(self.dtype, "bool"):
            if xp.any(xp.isinf(x)):
                raise ValueError(
                    f"Box unsigned int dtype don't support `np.inf`, x={x}"
                )

        # Cast integer infs to min and max of dtype
        arr = xp.zeros(self.shape, dtype=self.dtype, device=self.device)
        if xp.isdtype(self.dtype, "integral"):
            neg_inf, pos_inf = (xp.isinf(x)) & (x < 0), (xp.isinf(x)) & (x > 0)
            non_inf = ~neg_inf & ~pos_inf
            x = xpx.atleast_nd(x, ndim=arr.ndim, xp=xp)
            arr = xpx.at(arr, non_inf).set(x[non_inf])
            arr = xpx.at(arr, neg_inf).set(dtype_min)
            arr = xpx.at(arr, pos_inf).set(dtype_max)
        else:
            arr = xpx.at(arr, ...).set(x)

        if (
            is_array
            and xp.isdtype(self.dtype, "real floating")
            and xp.isdtype(x.dtype, "real floating")
            and xp.finfo(self.dtype).eps > xp.finfo(x.dtype).eps
        ):
            gym.logger.warn(
                f"Box low's precision lowered by casting to {self.dtype}, current dtype={x.dtype}"
            )

        # At this point, arr must have shape self.shape
        assert arr.shape == self.shape, f"Shape mismatch {arr.shape} != {self.shape}"
        bounded_below = ~xp.broadcast_to((xp.isinf(x)) & (x < 0), self.shape)
        bounded_above = ~xp.broadcast_to((xp.isinf(x)) & (x > 0), self.shape)
        return arr, bounded_below, bounded_above

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
        xp = array_namespace(self.low)
        below = bool(xp.all(self.bounded_below))
        above = bool(xp.all(self.bounded_above))
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

    def sample(self, mask: None = None, probability: None = None) -> NDArray[Any]:
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

        xp = array_namespace(self.low)
        is_float = xp.isdtype(self.dtype, "real floating")
        high = self.high if is_float else self.high.astype(xp.int64) + 1
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

        if not is_float:
            sample = xp.floor(sample)

        # clip values that would underflow/overflow
        if xp.isdtype(self.dtype, "signed integer"):
            dtype_min = xp.asarray(xp.iinfo(self.dtype).min + 2, dtype=sample.dtype)
            dtype_max = xp.asarray(xp.iinfo(self.dtype).max - 2, dtype=sample.dtype)
            sample = xp.clip(sample, min=dtype_min, max=dtype_max)
        elif xp.isdtype(self.dtype, "unsigned integer"):
            dtype_min = xp.asarray(xp.iinfo(self.dtype).min, dtype=sample.dtype)
            dtype_max = xp.asarray(xp.iinfo(self.dtype).max, dtype=sample.dtype)
            sample = xp.clip(sample, min=dtype_min, max=dtype_max)

        sample = xp.astype(sample, self.dtype)

        # float64 values have lower than integer precision near int64 min/max, so clip
        # again in case something has been cast to an out-of-bounds value
        if self.dtype == xp.int64:
            sample = xp.clip(sample, min=self.low, max=self.high)

        return sample

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        xp = array_namespace(self.low)
        if not isinstance(x, type(xp.empty(0))):
            gym.logger.warn("Casting input x to Array.")
            try:
                x = xp.asarray(x, dtype=self.dtype, device=self.device)
            except (ValueError, TypeError):
                return False

        return bool(
            xp.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and device(x) == self.device
            and xp.all(x >= self.low)
            and xp.all(x <= self.high)
        )

    def to_jsonable(self, sample_n: Sequence[Array]) -> list[list]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n: Sequence[float | int]) -> list[Array]:
        """Convert a JSONable data type to a batch of samples from this space."""
        xp = array_namespace(self.low)
        return [
            xp.asarray(sample, dtype=self.dtype, device=self.device)
            for sample in sample_n
        ]

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include bounds, shape and dtype.
        If a bound is uniform, only the corresponding scalar will be given to avoid redundant and ugly strings.

        Returns:
            A representation of the space
        """
        if not hasattr(self, "low"):
            return "Box(uninitialized)"
        return f"Box({array_short_repr(self.low)}, {array_short_repr(self.high)}, {self.shape}, {self.dtype}, {self.device})"

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        xp = array_namespace(self.low)
        return (
            isinstance(other, Box)
            and (self.shape == other.shape)
            and (self.dtype == other.dtype)
            and xp.allclose(self.low, other.low)
            and xp.allclose(self.high, other.high)
        )

    def _determine_dtype(
        self, dtype: DType | str | type | None, xp: ModuleType
    ) -> DType:
        if dtype is None:
            raise ValueError("Box dtype must be explicitly provided, cannot be None.")
        # The array API does not define something similar to np.dtype("float32"). We need to
        # implement this ourselves here.
        if isinstance(dtype, str):
            try:
                dtype = getattr(xp, dtype)
            except AttributeError:
                raise TypeError(f"data type '{dtype}' not understood")
        # Similarly, there is no xp.dtype(float) option, because some frameworks define the default
        # float dtype differently. We use the default float and integer dtypes for the framework.
        elif isinstance(dtype, type):
            if issubclass(dtype, float):
                dtype = xp.__array_namespace_info__().default_dtypes()["real floating"]
            elif issubclass(dtype, np.floating):
                ...  # Nothing to do here, dtype is passed through
            elif issubclass(dtype, bool) or issubclass(dtype, np.bool_):
                dtype = xp.bool
            elif issubclass(dtype, int):
                dtype = xp.__array_namespace_info__().default_dtypes()["integral"]
            elif issubclass(dtype, np.integer):
                ...  # Nothing to do here, dtype is passed through
        # If none of the above, it must already be a dtype

        #  * check that dtype is an accepted dtype
        try:
            if not any(xp.isdtype(dtype, kind=kind) for kind in self._dtype_kinds):
                raise ValueError(
                    f"Invalid Box dtype ({dtype}), must be an integer, floating, or bool dtype"
                )
        except TypeError:
            raise TypeError(f"Cannot interpret '{dtype}' as a data type")
        return xp.empty(0, dtype=dtype).dtype  # Array API doesn't have xp.dtype()

    def _determine_shape(
        self,
        low: SupportsFloat | Array,
        high: SupportsFloat | Array,
        shape: Sequence[int] | None,
        xp: ModuleType,
    ) -> tuple[int, ...]:
        # We can convert low and high because we already checked that they are convertible to Arrays
        low, high = xp.asarray(low), xp.asarray(high)
        if shape is None:
            try:
                low, high = xp.broadcast_arrays(low, high)
            except ValueError as e:
                raise ValueError(
                    f"Box low.shape and high.shape don't match, low.shape={low.shape}, high.shape={high.shape}"
                ) from e
            if low.shape == ():  # Single values
                return (1,)
            return low.shape
        if not isinstance(shape, Iterable):
            raise TypeError(
                f"Expected Box shape to be an iterable, actual type={type(shape)}"
            )
        if not all(isinstance(d, int) or isinstance(d, np.integer) for d in shape):
            raise TypeError(
                f"Expected all Box shape elements to be integer, actual type={tuple(type(dim) for dim in shape)}"
            )
        # Casts the `shape` argument to tuple[int, ...] (otherwise dim can `xp.int64`)
        shape = tuple(int(dim) for dim in shape)
        try:
            low = xp.broadcast_to(low, shape)
        except ValueError as e:
            raise ValueError(
                f"Box low.shape doesn't match provided shape, low.shape={low.shape}, shape={shape}"
            ) from e
        try:
            high = xp.broadcast_to(high, shape)
        except ValueError as e:
            raise ValueError(
                f"Box high.shape doesn't match provided shape, high.shape={high.shape}, shape={shape}"
            ) from e
        return shape

    def _determine_device(
        self, low: SupportsFloat | Array, high: SupportsFloat | Array, xp: ModuleType
    ) -> Device:
        """Determine the device of the space."""
        device1 = device(low) if is_array_api_obj(low) else None
        device2 = device(high) if is_array_api_obj(high) else None
        if device1 is not None and device2 is not None and device1 != device2:
            raise ValueError(
                f"Box low and high must be on the same device, got {device1} and {device2}"
            )
        if device1 is None and device2 is None:
            return device(xp.empty(0))  # Default device
        return device1 if device1 is not None else device2

    def _check_low_high(
        self, low: SupportsFloat | Array, high: SupportsFloat | Array, xp: ModuleType
    ) -> None:
        """Check if low and high are convertible to Arrays."""
        try:
            low = xp.asarray(low)
        except ValueError as e:
            raise ValueError(
                f"Box low must be an Array, integer, or float, actual type={type(low)}"
            ) from e
        try:
            high = xp.asarray(high)
        except ValueError as e:
            raise ValueError(
                f"Box high must be an Array, integer, or float, actual type={type(high)}"
            ) from e
        if not any(xp.isdtype(low.dtype, kind=kind) for kind in self._dtype_kinds):
            raise ValueError(
                f"Box low must be a floating, integer, or bool dtype, actual dtype={low.dtype}"
            )
        if not any(xp.isdtype(high.dtype, kind=kind) for kind in self._dtype_kinds):
            raise ValueError(
                f"Box high must be a floating, integer, or bool dtype, actual dtype={high.dtype}"
            )
