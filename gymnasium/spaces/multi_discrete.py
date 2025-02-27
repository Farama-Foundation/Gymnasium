"""Implementation of a space that represents the cartesian product of `Discrete` spaces."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.space import MaskNDArray, Space


class MultiDiscrete(Space[NDArray[np.integer]]):
    """This represents the cartesian product of arbitrary :class:`Discrete` spaces.

    It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space.

    Note:
        Some environment wrappers assume a value of 0 always represents the NOOP action.

    e.g. Nintendo Game Controller - Can be conceptualized as 3 discrete action spaces:

    1. Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
    2. Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    3. Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

    It can be initialized as ``MultiDiscrete([ 5, 2, 2 ])`` such that a sample might be ``array([3, 1, 0])``.

    Although this feature is rarely used, :class:`MultiDiscrete` spaces may also have several axes
    if ``nvec`` has several axes:

    Example:
        >>> from gymnasium.spaces import MultiDiscrete
        >>> import numpy as np
        >>> observation_space = MultiDiscrete(np.array([[1, 2], [3, 4]]), seed=42)
        >>> observation_space.sample()
        array([[0, 0],
               [2, 2]])
    """

    def __init__(
        self,
        nvec: NDArray[np.integer[Any]] | list[int],
        dtype: str | type[np.integer[Any]] = np.int64,
        seed: int | np.random.Generator | None = None,
        start: NDArray[np.integer[Any]] | list[int] | None = None,
    ):
        """Constructor of :class:`MultiDiscrete` space.

        The argument ``nvec`` will determine the number of values each categorical variable can take. If
        ``start`` is provided, it will define the minimal values corresponding to each categorical variable.

        Args:
            nvec: vector of counts of each categorical variable. This will usually be a list of integers. However,
                you may also pass a more complicated numpy array if you'd like the space to have several axes.
            dtype: This should be some kind of integer type.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
            start: Optionally, the starting value the element of each class will take (defaults to 0).
        """
        # determine dtype
        if dtype is None:
            raise ValueError(
                "MultiDiscrete dtype must be explicitly provided, cannot be None."
            )
        self.dtype = np.dtype(dtype)

        #  * check that dtype is an accepted dtype
        if not (np.issubdtype(self.dtype, np.integer)):
            raise ValueError(
                f"Invalid MultiDiscrete dtype ({self.dtype}), must be an integer dtype"
            )

        self.nvec = np.array(nvec, dtype=dtype, copy=True)
        if start is not None:
            self.start = np.array(start, dtype=dtype, copy=True)
        else:
            self.start = np.zeros(self.nvec.shape, dtype=dtype)

        assert (
            self.start.shape == self.nvec.shape
        ), "start and nvec (counts) should have the same shape"
        assert (self.nvec > 0).all(), "nvec (counts) have to be positive"

        super().__init__(self.nvec.shape, self.dtype, seed)

    @property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than :class:`gym.Space` - never None."""
        return self._shape  # type: ignore

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def sample(
        self,
        mask: tuple[MaskNDArray, ...] | None = None,
        probability: tuple[MaskNDArray, ...] | None = None,
    ) -> NDArray[np.integer[Any]]:
        """Generates a single random sample from this space.

        Args:
            mask: An optional mask for multi-discrete, expects tuples with a ``np.ndarray`` mask in the position of each
                action with shape ``(n,)`` where ``n`` is the number of actions and ``dtype=np.int8``.
                Only ``mask values == 1`` are possible to sample unless all mask values for an action are ``0`` then the default action ``self.start`` (the smallest element) is sampled.
            probability: An optional probability mask for multi-discrete, expects tuples with a ``np.ndarray`` probability mask in the position of each
                action with shape ``(n,)`` where ``n`` is the number of actions and ``dtype=np.float64``.
                Only probability mask values within ``[0,1]`` are possible to sample as long as the sum of all values is ``1``.

        Returns:
            An ``np.ndarray`` of :meth:`Space.shape`
        """
        if mask is not None and probability is not None:
            raise ValueError(
                f"Only one of `mask` or `probability` can be provided, actual values: mask={mask}, probability={probability}"
            )
        elif mask is not None:
            return np.array(
                self._apply_mask(mask, self.nvec, self.start, "mask"),
                dtype=self.dtype,
            )
        elif probability is not None:
            return np.array(
                self._apply_mask(probability, self.nvec, self.start, "probability"),
                dtype=self.dtype,
            )
        else:
            return (self.np_random.random(self.nvec.shape) * self.nvec).astype(
                self.dtype
            ) + self.start

    def _apply_mask(
        self,
        sub_mask: MaskNDArray | tuple[MaskNDArray, ...],
        sub_nvec: MaskNDArray | np.integer[Any],
        sub_start: MaskNDArray | np.integer[Any],
        mask_type: str,
    ) -> int | list[Any]:
        """Returns a sample using the provided mask or probability mask."""
        if isinstance(sub_nvec, np.ndarray):
            assert isinstance(
                sub_mask, tuple
            ), f"Expects the mask to be a tuple for sub_nvec ({sub_nvec}), actual type: {type(sub_mask)}"
            assert len(sub_mask) == len(
                sub_nvec
            ), f"Expects the mask length to be equal to the number of actions, mask length: {len(sub_mask)}, nvec length: {len(sub_nvec)}"
            return [
                self._apply_mask(new_mask, new_nvec, new_start, mask_type)
                for new_mask, new_nvec, new_start in zip(sub_mask, sub_nvec, sub_start)
            ]

        assert np.issubdtype(
            type(sub_nvec), np.integer
        ), f"Expects the sub_nvec to be an action, actually: {sub_nvec}, {type(sub_nvec)}"
        assert isinstance(
            sub_mask, np.ndarray
        ), f"Expects the sub mask to be np.ndarray, actual type: {type(sub_mask)}"
        assert (
            len(sub_mask) == sub_nvec
        ), f"Expects the mask length to be equal to the number of actions, mask length: {len(sub_mask)}, action: {sub_nvec}"

        if mask_type == "mask":
            assert (
                sub_mask.dtype == np.int8
            ), f"Expects the mask dtype to be np.int8, actual dtype: {sub_mask.dtype}"

            valid_action_mask = sub_mask == 1
            assert np.all(
                np.logical_or(sub_mask == 0, valid_action_mask)
            ), f"Expects all masks values to 0 or 1, actual values: {sub_mask}"

            if np.any(valid_action_mask):
                return self.np_random.choice(np.where(valid_action_mask)[0]) + sub_start
            else:
                return sub_start
        elif mask_type == "probability":
            assert (
                sub_mask.dtype == np.float64
            ), f"Expects the mask dtype to be np.float64, actual dtype: {sub_mask.dtype}"
            valid_action_mask = np.logical_and(sub_mask > 0, sub_mask <= 1)
            assert np.all(
                np.logical_or(sub_mask == 0, valid_action_mask)
            ), f"Expects all masks values to be between 0 and 1, actual values: {sub_mask}"
            assert np.isclose(
                np.sum(sub_mask), 1
            ), f"Expects the sum of all mask values to be 1, actual sum: {np.sum(sub_mask)}"

            normalized_sub_mask = sub_mask / np.sum(sub_mask)
            return (
                self.np_random.choice(
                    np.where(valid_action_mask)[0],
                    p=normalized_sub_mask[valid_action_mask],
                )
                + sub_start
            )
        raise ValueError(f"Unsupported mask type: {mask_type}")

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check

        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return bool(
            isinstance(x, np.ndarray)
            and x.shape == self.shape
            and x.dtype != object
            and np.all(self.start <= x)
            and np.all(x - self.start < self.nvec)
        )

    def to_jsonable(
        self, sample_n: Sequence[NDArray[np.integer[Any]]]
    ) -> list[Sequence[int]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(
        self, sample_n: list[Sequence[int]]
    ) -> list[NDArray[np.integer[Any]]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [np.array(sample, dtype=np.int64) for sample in sample_n]

    def __repr__(self):
        """Gives a string representation of this space."""
        if np.any(self.start != 0):
            return f"MultiDiscrete({self.nvec}, start={self.start})"
        return f"MultiDiscrete({self.nvec})"

    def __getitem__(self, index: int | tuple[int, ...]):
        """Extract a subspace from this ``MultiDiscrete`` space."""
        nvec = self.nvec[index]
        start = self.start[index]
        if nvec.ndim == 0:
            subspace = Discrete(nvec, start=start)
        else:
            subspace = MultiDiscrete(nvec, self.dtype, start=start)

        # you don't need to deepcopy as np random generator call replaces the state not the data
        subspace.np_random.bit_generator.state = self.np_random.bit_generator.state

        return subspace

    def __len__(self):
        """Gives the ``len`` of samples from this space."""
        if self.nvec.ndim >= 2:
            gym.logger.warn(
                "Getting the length of a multi-dimensional MultiDiscrete space."
            )
        return len(self.nvec)

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return bool(
            isinstance(other, MultiDiscrete)
            and self.dtype == other.dtype
            and self.shape == other.shape
            and np.all(self.nvec == other.nvec)
            and np.all(self.start == other.start)
        )

    def __setstate__(self, state: Iterable[tuple[str, Any]] | Mapping[str, Any]):
        """Used when loading a pickled space.

        This method has to be implemented explicitly to allow for loading of legacy states.

        Args:
            state: The new state
        """
        state = dict(state)

        if "start" not in state:
            state["start"] = np.zeros(state["_shape"], dtype=state["dtype"])

        super().__setstate__(state)
