"""Implementation of a space that represents the cartesian product of other spaces."""

from __future__ import annotations

import typing
from collections.abc import Iterable
from typing import Any

import numpy as np

from gymnasium.spaces.space import Space


class Tuple(Space[tuple[Any, ...]], typing.Sequence[Any]):
    """A tuple (more precisely: the cartesian product) of :class:`Space` instances.

    Elements of this space are tuples of elements of the constituent spaces.

    Example:
        >>> from gymnasium.spaces import Tuple, Box, Discrete
        >>> observation_space = Tuple((Discrete(2), Box(-1, 1, shape=(2,))), seed=42)
        >>> observation_space.sample()
        (np.int64(0), array([-0.3991573 ,  0.21649833], dtype=float32))
    """

    def __init__(
        self,
        spaces: Iterable[Space[Any]],
        seed: int | typing.Sequence[int] | np.random.Generator | None = None,
    ):
        r"""Constructor of :class:`Tuple` space.

        The generated instance will represent the cartesian product :math:`\text{spaces}[0] \times ... \times \text{spaces}[-1]`.

        Args:
            spaces (Iterable[Space]): The spaces that are involved in the cartesian product.
            seed: Optionally, you can use this argument to seed the RNGs of the ``spaces`` to ensure reproducible sampling.
        """
        self.spaces = tuple(spaces)
        for space in self.spaces:
            assert isinstance(
                space, Space
            ), f"{space} does not inherit from `gymnasium.Space`. Actual Type: {type(space)}"
        super().__init__(None, None, seed)  # type: ignore

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return all(space.is_np_flattenable for space in self.spaces)

    def seed(self, seed: int | typing.Sequence[int] | None = None) -> tuple[int, ...]:
        """Seed the PRNG of this space and all subspaces.

        Depending on the type of seed, the subspaces will be seeded differently

        * ``None`` - All the subspaces will use a random initial seed
        * ``Int`` - The integer is used to seed the :class:`Tuple` space that is used to generate seed values for each of the subspaces. Warning, this does not guarantee unique seeds for all the subspaces.
        * ``List`` / ``Tuple`` - Values used to seed the subspaces. This allows the seeding of multiple composite subspaces ``[42, 54, ...]``.

        Args:
            seed: An optional list of ints or int to seed the (sub-)spaces.

        Returns:
            A tuple of the seed values for all subspaces
        """
        if seed is None:
            return tuple(space.seed(None) for space in self.spaces)
        elif isinstance(seed, int):
            super().seed(seed)
            subseeds = self.np_random.integers(
                np.iinfo(np.int32).max, size=len(self.spaces)
            )
            return tuple(
                subspace.seed(int(subseed))
                for subspace, subseed in zip(self.spaces, subseeds)
            )
        elif isinstance(seed, (tuple, list)):
            if len(seed) != len(self.spaces):
                raise ValueError(
                    f"Expects that the subspaces of seeds equals the number of subspaces. Actual length of seeds: {len(seed)}, length of subspaces: {len(self.spaces)}"
                )

            return tuple(
                space.seed(subseed) for subseed, space in zip(seed, self.spaces)
            )
        else:
            raise TypeError(
                f"Expected seed type: list, tuple, int or None, actual type: {type(seed)}"
            )

    def sample(
        self,
        mask: tuple[Any | None, ...] | None = None,
        probability: tuple[Any | None, ...] | None = None,
    ) -> tuple[Any, ...]:
        """Generates a single random sample inside this space.

        This method draws independent samples from the subspaces.

        Args:
            mask: An optional tuple of optional masks for each of the subspace's samples,
                expects the same number of masks as spaces
            probability: An optional tuple of optional probability masks for each of the subspace's samples,
                expects the same number of probability masks as spaces

        Returns:
            Tuple of the subspace's samples
        """
        if mask is not None and probability is not None:
            raise ValueError(
                f"Only one of `mask` or `probability` can be provided, actual values: mask={mask}, probability={probability}"
            )
        elif mask is not None:
            assert isinstance(
                mask, tuple
            ), f"Expected type of `mask` to be tuple, actual type: {type(mask)}"
            assert len(mask) == len(
                self.spaces
            ), f"Expected length of `mask` to be {len(self.spaces)}, actual length: {len(mask)}"

            return tuple(
                space.sample(mask=space_mask)
                for space, space_mask in zip(self.spaces, mask)
            )

        elif probability is not None:
            assert isinstance(
                probability, tuple
            ), f"Expected type of `probability` to be tuple, actual type: {type(probability)}"
            assert len(probability) == len(
                self.spaces
            ), f"Expected length of `probability` to be {len(self.spaces)}, actual length: {len(probability)}"

            return tuple(
                space.sample(probability=space_probability)
                for space, space_probability in zip(self.spaces, probability)
            )
        else:
            return tuple(space.sample() for space in self.spaces)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, (list, np.ndarray)):
            x = tuple(x)  # Promote list and ndarray to tuple for contains check

        return (
            isinstance(x, tuple)
            and len(x) == len(self.spaces)
            and all(space.contains(part) for (space, part) in zip(self.spaces, x))
        )

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return "Tuple(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def to_jsonable(
        self, sample_n: typing.Sequence[tuple[Any, ...]]
    ) -> list[list[Any]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        # serialize as list-repr of tuple of vectors
        return [
            space.to_jsonable([sample[i] for sample in sample_n])
            for i, space in enumerate(self.spaces)
        ]

    def from_jsonable(self, sample_n: list[list[Any]]) -> list[tuple[Any, ...]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [
            sample
            for sample in zip(
                *[
                    space.from_jsonable(sample_n[i])
                    for i, space in enumerate(self.spaces)
                ]
            )
        ]

    def __getitem__(self, index: int) -> Space[Any]:
        """Get the subspace at specific `index`."""
        return self.spaces[index]

    def __len__(self) -> int:
        """Get the number of subspaces that are involved in the cartesian product."""
        return len(self.spaces)

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return isinstance(other, Tuple) and self.spaces == other.spaces
