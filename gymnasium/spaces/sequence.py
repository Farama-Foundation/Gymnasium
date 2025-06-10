"""Implementation of a space that represents finite-length sequences."""

from __future__ import annotations

import typing
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium.spaces.space import Space


class Sequence(Space[Union[tuple[Any, ...], Any]]):
    r"""This space represent sets of finite-length sequences.

    This space represents the set of tuples of the form :math:`(a_0, \dots, a_n)` where the :math:`a_i` belong
    to some space that is specified during initialization and the integer :math:`n` is not fixed

    Example:
        >>> from gymnasium.spaces import Sequence, Box
        >>> observation_space = Sequence(Box(0, 1), seed=0)
        >>> observation_space.sample()
        (array([0.6822636], dtype=float32), array([0.18933342], dtype=float32), array([0.19049619], dtype=float32))
        >>> observation_space.sample()
        (array([0.83506], dtype=float32), array([0.9053838], dtype=float32), array([0.5836242], dtype=float32), array([0.63214064], dtype=float32))

    Example with stacked observations
        >>> observation_space = Sequence(Box(0, 1), stack=True, seed=0)
        >>> observation_space.sample()
        array([[0.6822636 ],
               [0.18933342],
               [0.19049619]], dtype=float32)
    """

    def __init__(
        self,
        space: Space[Any],
        seed: int | np.random.Generator | None = None,
        stack: bool = False,
    ):
        """Constructor of the :class:`Sequence` space.

        Args:
            space: Elements in the sequences this space represent must belong to this space.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
            stack: If ``True`` then the resulting samples would be stacked.
        """
        assert isinstance(
            space, Space
        ), f"Expects the feature space to be instance of a gym Space, actual type: {type(space)}"
        self.feature_space = space
        self.stack = stack
        if self.stack:
            self.stacked_feature_space: Space = gym.vector.utils.batch_space(
                self.feature_space, 1
            )

        # None for shape and dtype, since it'll require special handling
        super().__init__(None, None, seed)

    def seed(self, seed: int | tuple[int, int] | None = None) -> tuple[int, int]:
        """Seed the PRNG of the Sequence space and the feature space.

        Depending on the type of seed, the subspaces will be seeded differently

        * ``None`` - All the subspaces will use a random initial seed
        * ``Int`` - The integer is used to seed the :class:`Sequence` space that is used to generate a seed value for the feature space.
        * ``Tuple of ints`` - A tuple for the :class:`Sequence` and feature space.

        Args:
            seed: An optional int or tuple of ints to seed the PRNG. See above for more details

        Returns:
            A tuple of the seeding values for the Sequence and feature space
        """
        if seed is None:
            return super().seed(None), self.feature_space.seed(None)
        elif isinstance(seed, int):
            super_seed = super().seed(seed)
            feature_seed = int(self.np_random.integers(np.iinfo(np.int32).max))
            # this is necessary such that after int or list/tuple seeding, the Sequence PRNG are equivalent
            super().seed(seed)
            return super_seed, self.feature_space.seed(feature_seed)
        elif isinstance(seed, (tuple, list)):
            if len(seed) != 2:
                raise ValueError(
                    f"Expects the seed to have two elements for the Sequence and feature space, actual length: {len(seed)}"
                )
            return super().seed(seed[0]), self.feature_space.seed(seed[1])
        else:
            raise TypeError(
                f"Expected None, int, tuple of ints, actual type: {type(seed)}"
            )

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return False

    def sample(
        self,
        mask: None | (
            tuple[
                None | int | NDArray[np.integer],
                Any,
            ]
        ) = None,
        probability: None | (
            tuple[
                None | int | NDArray[np.integer],
                Any,
            ]
        ) = None,
    ) -> tuple[Any] | Any:
        """Generates a single random sample from this space.

        Args:
            mask: An optional mask for (optionally) the length of the sequence and (optionally) the values in the sequence.
                If you specify ``mask``, it is expected to be a tuple of the form ``(length_mask, sample_mask)`` where ``length_mask`` is

                * ``None`` - The length will be randomly drawn from a geometric distribution
                * ``int`` - Fixed length
                * ``np.ndarray`` of integers - Length of the sampled sequence is randomly drawn from this array.

                The second element of the tuple ``sample_mask`` specifies how the feature space will be sampled.
                Depending on if mask or probability is used will affect what argument is used.
            probability: See mask description above, the only difference is on the ``sample_mask`` for the feature space being probability rather than mask.

        Returns:
            A tuple of random length with random samples of elements from the :attr:`feature_space`.
        """
        if mask is not None and probability is not None:
            raise ValueError(
                f"Only one of `mask` or `probability` can be provided, actual values: mask={mask}, probability={probability}"
            )
        elif mask is not None:
            sample_length = self.generate_sample_length(mask[0], "mask")
            sampled_values = tuple(
                self.feature_space.sample(mask=mask[1]) for _ in range(sample_length)
            )
        elif probability is not None:
            sample_length = self.generate_sample_length(probability[0], "probability")
            sampled_values = tuple(
                self.feature_space.sample(probability=probability[1])
                for _ in range(sample_length)
            )
        else:
            sample_length = self.np_random.geometric(0.25)
            sampled_values = tuple(
                self.feature_space.sample() for _ in range(sample_length)
            )

        if self.stack:
            # Concatenate values if stacked.
            out = gym.vector.utils.create_empty_array(
                self.feature_space, len(sampled_values)
            )
            return gym.vector.utils.concatenate(self.feature_space, sampled_values, out)

        return sampled_values

    def generate_sample_length(
        self,
        length_mask: None | np.integer | NDArray[np.integer],
        mask_type: None | str,
    ) -> int:
        """Generate the sample length for a given length mask and mask type."""
        if length_mask is not None:
            if np.issubdtype(type(length_mask), np.integer):
                assert (
                    0 <= length_mask
                ), f"Expects the length mask of `{mask_type}` to be greater than or equal to zero, actual value: {length_mask}"

                return length_mask
            elif isinstance(length_mask, np.ndarray):
                assert (
                    len(length_mask.shape) == 1
                ), f"Expects the shape of the length mask of `{mask_type}` to be 1-dimensional, actual shape: {length_mask.shape}"
                assert np.all(
                    0 <= length_mask
                ), f"Expects all values in the length_mask of `{mask_type}` to be greater than or equal to zero, actual values: {length_mask}"
                assert np.issubdtype(
                    length_mask.dtype, np.integer
                ), f"Expects the length mask array of `{mask_type}` to have dtype of np.integer, actual type: {length_mask.dtype}"

                return self.np_random.choice(length_mask)
            else:
                raise TypeError(
                    f"Expects the type of length_mask of `{mask_type}` to be an integer or a np.ndarray, actual type: {type(length_mask)}"
                )
        else:
            # The choice of 0.25 is arbitrary
            return self.np_random.geometric(0.25)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        # by definition, any sequence is an iterable
        if self.stack:
            return all(
                item in self.feature_space
                for item in gym.vector.utils.iterate(self.stacked_feature_space, x)
            )
        else:
            return isinstance(x, tuple) and all(
                self.feature_space.contains(item) for item in x
            )

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"Sequence({self.feature_space}, stack={self.stack})"

    def to_jsonable(
        self, sample_n: typing.Sequence[tuple[Any, ...] | Any]
    ) -> list[list[Any]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        if self.stack:
            return self.stacked_feature_space.to_jsonable(sample_n)
        else:
            return [self.feature_space.to_jsonable(sample) for sample in sample_n]

    def from_jsonable(self, sample_n: list[list[Any]]) -> list[tuple[Any, ...] | Any]:
        """Convert a JSONable data type to a batch of samples from this space."""
        if self.stack:
            return self.stacked_feature_space.from_jsonable(sample_n)
        else:
            return [
                tuple(self.feature_space.from_jsonable(sample)) for sample in sample_n
            ]

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Sequence)
            and self.feature_space == other.feature_space
            and self.stack == other.stack
        )
