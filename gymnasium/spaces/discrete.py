"""Implementation of a space consisting of finitely many elements."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from gymnasium.spaces.space import MaskNDArray, Space


class Discrete(Space[np.int64]):
    r"""A space consisting of finitely many elements.

    This class represents a finite subset of integers, more specifically a set of the form :math:`\{ a, a+1, \dots, a+n-1 \}`.

    Example:
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
    """

    def __init__(
        self,
        n: int | np.integer[Any],
        seed: int | np.random.Generator | None = None,
        start: int | np.integer[Any] = 0,
    ):
        r"""Constructor of :class:`Discrete` space.

        This will construct the space :math:`\{\text{start}, ..., \text{start} + n - 1\}`.

        Args:
            n (int): The number of elements of this space.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the ``Dict`` space.
            start (int): The smallest element of this space.
        """
        assert np.issubdtype(
            type(n), np.integer
        ), f"Expects `n` to be an integer, actual dtype: {type(n)}"
        assert n > 0, "n (counts) have to be positive"
        assert np.issubdtype(
            type(start), np.integer
        ), f"Expects `start` to be an integer, actual type: {type(start)}"

        self.n = np.int64(n)
        self.start = np.int64(start)
        super().__init__((), np.int64, seed)

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def sample(
        self, mask: MaskNDArray | None = None, probability: MaskNDArray | None = None
    ) -> np.int64:
        """Generates a single random sample from this space.

        A sample will be chosen uniformly at random with the mask if provided, or it will be chosen according to a specified probability distribution if the probability mask is provided.

        Args:
            mask: An optional mask for if an action can be selected.
                Expected `np.ndarray` of shape ``(n,)`` and dtype ``np.int8`` where ``1`` represents valid actions and ``0`` invalid / infeasible actions.
                If there are no possible actions (i.e. ``np.all(mask == 0)``) then ``space.start`` will be returned.
            probability: An optional probability mask describing the probability of each action being selected.
                Expected `np.ndarray` of shape ``(n,)`` and dtype ``np.float64`` where each value is in the range ``[0, 1]`` and the sum of all values is 1.
                If the values do not sum to 1, an exception will be thrown.

        Returns:
            A sampled integer from the space
        """
        if mask is not None and probability is not None:
            raise ValueError(
                f"Only one of `mask` or `probability` can be provided, actual values: mask={mask}, probability={probability}"
            )
        # binary mask sampling
        elif mask is not None:
            assert isinstance(
                mask, np.ndarray
            ), f"The expected type of the sample mask is np.ndarray, actual type: {type(mask)}"
            assert (
                mask.dtype == np.int8
            ), f"The expected dtype of the sample mask is np.int8, actual dtype: {mask.dtype}"
            assert mask.shape == (
                self.n,
            ), f"The expected shape of the sample mask is {(int(self.n),)}, actual shape: {mask.shape}"

            valid_action_mask = mask == 1
            assert np.all(
                np.logical_or(mask == 0, valid_action_mask)
            ), f"All values of the sample mask should be 0 or 1, actual values: {mask}"

            if np.any(valid_action_mask):
                return self.start + self.np_random.choice(
                    np.where(valid_action_mask)[0]
                )
            else:
                return self.start
        # probability mask sampling
        elif probability is not None:
            assert isinstance(
                probability, np.ndarray
            ), f"The expected type of the sample probability is np.ndarray, actual type: {type(probability)}"
            assert (
                probability.dtype == np.float64
            ), f"The expected dtype of the sample probability is np.float64, actual dtype: {probability.dtype}"
            assert probability.shape == (
                self.n,
            ), f"The expected shape of the sample probability is {(int(self.n),)}, actual shape: {probability.shape}"

            assert np.all(
                np.logical_and(probability >= 0, probability <= 1)
            ), f"All values of the sample probability should be between 0 and 1, actual values: {probability}"
            assert np.isclose(
                np.sum(probability), 1
            ), f"The sum of the sample probability should be equal to 1, actual sum: {np.sum(probability)}"

            return self.start + self.np_random.choice(np.arange(self.n), p=probability)
        # uniform sampling
        else:
            return self.start + self.np_random.integers(self.n)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, int):
            as_int64 = np.int64(x)
        elif isinstance(x, (np.generic, np.ndarray)) and (
            np.issubdtype(x.dtype, np.integer) and x.shape == ()
        ):
            as_int64 = np.int64(x)
        else:
            return False

        return bool(self.start <= as_int64 < self.start + self.n)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        if self.start != 0:
            return f"Discrete({self.n}, start={self.start})"
        return f"Discrete({self.n})"

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )

    def __setstate__(self, state: Iterable[tuple[str, Any]] | Mapping[str, Any]):
        """Used when loading a pickled space.

        This method has to be implemented explicitly to allow for loading of legacy states.

        Args:
            state: The new state
        """
        # Don't mutate the original state
        state = dict(state)

        # Allow for loading of legacy states.
        # See https://github.com/openai/gym/pull/2470
        if "start" not in state:
            state["start"] = np.int64(0)

        super().__setstate__(state)

    def to_jsonable(self, sample_n: Sequence[np.int64]) -> list[int]:
        """Converts a list of samples to a list of ints."""
        return [int(x) for x in sample_n]

    def from_jsonable(self, sample_n: list[int]) -> list[np.int64]:
        """Converts a list of json samples to a list of np.int64."""
        return [np.int64(x) for x in sample_n]
