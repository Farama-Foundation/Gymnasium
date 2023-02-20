"""Implementation of a space that represents the cartesian product of other spaces."""
from __future__ import annotations

import collections.abc
import typing
from typing import Any, Iterable

import numpy as np

from gymnasium.spaces.space import Space


class Tuple(Space[typing.Tuple[Any, ...]], typing.Sequence[Any]):
    """A tuple (more precisely: the cartesian product) of :class:`Space` instances.

    Elements of this space are tuples of elements of the constituent spaces.

    Example:
        >>> from gymnasium.spaces import Tuple, Box, Discrete
        >>> observation_space = Tuple((Discrete(2), Box(-1, 1, shape=(2,))), seed=42)
        >>> observation_space.sample()
        (0, array([-0.3991573 ,  0.21649833], dtype=float32))
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

    def seed(self, seed: int | typing.Sequence[int] | None = None) -> list[int]:
        """Seed the PRNG of this space and all subspaces.

        Depending on the type of seed, the subspaces will be seeded differently

        * ``None`` - All the subspaces will use a random initial seed
        * ``Int`` - The integer is used to seed the `Tuple` space that is used to generate seed values for each of the subspaces. Warning, this does not guarantee unique seeds for all of the subspaces.
        * ``List`` - Values used to seed the subspaces. This allows the seeding of multiple composite subspaces (``List(42, 54, ...``).

        Args:
            seed: An optional list of ints or int to seed the (sub-)spaces.
        """
        seeds: list[int] = []

        if isinstance(seed, collections.abc.Sequence):
            assert len(seed) == len(
                self.spaces
            ), f"Expects that the subspaces of seeds equals the number of subspaces. Actual length of seeds: {len(seeds)}, length of subspaces: {len(self.spaces)}"
            for subseed, space in zip(seed, self.spaces):
                seeds += space.seed(subseed)
        elif isinstance(seed, int):
            seeds = super().seed(seed)
            subseeds = self.np_random.integers(
                np.iinfo(np.int32).max, size=len(self.spaces)
            )
            for subspace, subseed in zip(self.spaces, subseeds):
                seeds += subspace.seed(int(subseed))
        elif seed is None:
            for space in self.spaces:
                seeds += space.seed(seed)
        else:
            raise TypeError(
                f"Expected seed type: list, tuple, int or None, actual type: {type(seed)}"
            )

        return seeds

    def sample(self, mask: tuple[Any | None, ...] | None = None) -> tuple[Any, ...]:
        """Generates a single random sample inside this space.

        This method draws independent samples from the subspaces.

        Args:
            mask: An optional tuple of optional masks for each of the subspace's samples,
                expects the same number of masks as spaces

        Returns:
            Tuple of the subspace's samples
        """
        if mask is not None:
            assert isinstance(
                mask, tuple
            ), f"Expected type of mask is tuple, actual type: {type(mask)}"
            assert len(mask) == len(
                self.spaces
            ), f"Expected length of mask is {len(self.spaces)}, actual length: {len(mask)}"

            return tuple(
                space.sample(mask=sub_mask)
                for space, sub_mask in zip(self.spaces, mask)
            )

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
