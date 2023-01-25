"""Implementation of a space that represents finite-length sequences."""
from __future__ import annotations

import collections.abc
import typing
from typing import Any

import numpy as np
import numpy.typing as npt

from gymnasium.spaces.space import Space


class Sequence(Space[typing.Tuple[Any, ...]]):
    r"""This space represent sets of finite-length sequences.

    This space represents the set of tuples of the form :math:`(a_0, \dots, a_n)` where the :math:`a_i` belong
    to some space that is specified during initialization and the integer :math:`n` is not fixed

    Example::
        >>> from gymnasium.spaces import Box
        >>> space = Sequence(Box(0, 1), seed=42)
        >>> space.sample()   # doctest: +SKIP
        (array([0.6369617], dtype=float32),)
        >>> space.sample()   # doctest: +SKIP
        (array([0.01652764], dtype=float32), array([0.8132702], dtype=float32),)
    """

    def __init__(
        self,
        space: Space[Any],
        seed: int | np.random.Generator | None = None,
    ):
        """Constructor of the :class:`Sequence` space.

        Args:
            space: Elements in the sequences this space represent must belong to this space.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
        """
        assert isinstance(
            space, Space
        ), f"Expects the feature space to be instance of a gym Space, actual type: {type(space)}"
        self.feature_space = space

        # None for shape and dtype, since it'll require special handling
        super().__init__(None, None, seed)

    def seed(self, seed: int | None = None) -> list[int]:
        """Seed the PRNG of this space and the feature space."""
        seeds = super().seed(seed)
        seeds += self.feature_space.seed(seed)
        return seeds

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return False

    def sample(
        self,
        mask: None
        | (
            tuple[
                None | np.integer | npt.NDArray[np.integer],
                Any,
            ]
        ) = None,
    ) -> tuple[Any]:
        """Generates a single random sample from this space.

        Args:
            mask: An optional mask for (optionally) the length of the sequence and (optionally) the values in the sequence.
                If you specify `mask`, it is expected to be a tuple of the form `(length_mask, sample_mask)` where `length_mask`
                is

                * ``None`` The length will be randomly drawn from a geometric distribution
                * ``np.ndarray`` of integers, in which case the length of the sampled sequence is randomly drawn from this array.
                * ``int`` for a fixed length sample

                The second element of the mask tuple `sample` mask specifies a mask that is applied when
                sampling elements from the base space. The mask is applied for each feature space sample.

        Returns:
            A tuple of random length with random samples of elements from the :attr:`feature_space`.
        """
        if mask is not None:
            length_mask, feature_mask = mask
        else:
            length_mask, feature_mask = None, None

        if length_mask is not None:
            if np.issubdtype(type(length_mask), np.integer):
                assert (
                    0 <= length_mask
                ), f"Expects the length mask to be greater than or equal to zero, actual value: {length_mask}"
                length = length_mask
            elif isinstance(length_mask, np.ndarray):
                assert (
                    len(length_mask.shape) == 1
                ), f"Expects the shape of the length mask to be 1-dimensional, actual shape: {length_mask.shape}"
                assert np.all(
                    0 <= length_mask
                ), f"Expects all values in the length_mask to be greater than or equal to zero, actual values: {length_mask}"
                assert np.issubdtype(
                    length_mask.dtype, np.integer
                ), f"Expects the length mask array to have dtype to be an numpy integer, actual type: {length_mask.dtype}"
                length = self.np_random.choice(length_mask)
            else:
                raise TypeError(
                    f"Expects the type of length_mask to an integer or a np.ndarray, actual type: {type(length_mask)}"
                )
        else:
            # The choice of 0.25 is arbitrary
            length = self.np_random.geometric(0.25)

        return tuple(
            self.feature_space.sample(mask=feature_mask) for _ in range(length)
        )

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        # by definition, any sequence is an iterable
        return isinstance(x, collections.abc.Iterable) and all(
            self.feature_space.contains(item) for item in x
        )

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"Sequence({self.feature_space})"

    def to_jsonable(
        self, sample_n: typing.Sequence[tuple[Any, ...]]
    ) -> list[list[Any]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        # serialize as dict-repr of vectors
        return [self.feature_space.to_jsonable(list(sample)) for sample in sample_n]

    def from_jsonable(self, sample_n: list[list[Any]]) -> list[tuple[Any, ...]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [tuple(self.feature_space.from_jsonable(sample)) for sample in sample_n]

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return isinstance(other, Sequence) and self.feature_space == other.feature_space
