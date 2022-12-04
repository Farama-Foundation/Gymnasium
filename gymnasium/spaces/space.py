"""Implementation of the `Space` metaclass."""
from __future__ import annotations

from typing import Any, Generic, Iterable, Mapping, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

from gymnasium.utils import seeding


T_cov = TypeVar("T_cov", covariant=True)


MaskNDArray = npt.NDArray[np.int8]


class Space(Generic[T_cov]):
    """Superclass that is used to define observation and action spaces.

    Spaces are crucially used in Gym to define the format of valid actions and observations.
    They serve various purposes:

    * They clearly define how to interact with environments, i.e. they specify what actions need to look like
      and what observations will look like
    * They allow us to work with highly structured data (e.g. in the form of elements of :class:`Dict` spaces)
      and painlessly transform them into flat arrays that can be used in learning code
    * They provide a method to sample random elements. This is especially useful for exploration and debugging.

    Different spaces can be combined hierarchically via container spaces (:class:`Tuple` and :class:`Dict`) to build a
    more expressive space

    Warning:
        Custom observation & action spaces can inherit from the ``Space``
        class. However, most use-cases should be covered by the existing space
        classes (e.g. :class:`Box`, :class:`Discrete`, etc...), and container classes (:class`Tuple` &
        :class:`Dict`). Note that parametrized probability distributions (through the
        :meth:`Space.sample()` method), and batching functions (in :class:`gym.vector.VectorEnv`), are
        only well-defined for instances of spaces provided in gym by default.
        Moreover, some implementations of Reinforcement Learning algorithms might
        not handle custom spaces properly. Use custom spaces with care.
    """

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        dtype: npt.DTypeLike | None = None,
        seed: int | np.random.Generator | None = None,
    ):
        """Constructor of :class:`Space`.

        Args:
            shape (Optional[Sequence[int]]): If elements of the space are numpy arrays, this should specify their shape.
            dtype (Optional[Type | str]): If elements of the space are numpy arrays, this should specify their dtype.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space
        """
        self._shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._np_random = None
        if seed is not None:
            if isinstance(seed, np.random.Generator):
                self._np_random = seed
            else:
                self.seed(seed)

    @property
    def np_random(self) -> np.random.Generator:
        """Lazily seed the PRNG since this is expensive and only needed if sampling from this space.

        As :meth:`seed` is not guaranteed to set the `_np_random` for particular seeds. We add a
        check after :meth:`seed` to set a new random number generator.
        """
        if self._np_random is None:
            self.seed()

        # As `seed` is not guaranteed (in particular for composite spaces) to set the `_np_random` then we set it randomly.
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()

        return self._np_random

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Return the shape of the space as an immutable property."""
        return self._shape

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`."""
        raise NotImplementedError

    def sample(self, mask: Any | None = None) -> T_cov:
        """Randomly sample an element of this space.

        Can be uniform or non-uniform sampling based on boundedness of space.

        Args:
            mask: A mask used for sampling, expected ``dtype=np.int8`` and see sample implementation for expected shape.

        Returns:
            A sampled actions from the space
        """
        raise NotImplementedError

    def seed(self, seed: int | None = None) -> list[int]:
        """Seed the PRNG of this space and possibly the PRNGs of subspaces."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        raise NotImplementedError

    def __contains__(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return self.contains(x)

    def __setstate__(self, state: Iterable[tuple[str, Any]] | Mapping[str, Any]):
        """Used when loading a pickled space.

        This method was implemented explicitly to allow for loading of legacy states.

        Args:
            state: The updated state value
        """
        # Don't mutate the original state
        state = dict(state)

        # Allow for loading of legacy states.
        # See:
        #   https://github.com/openai/gym/pull/2397 -- shape
        #   https://github.com/openai/gym/pull/1913 -- np_random
        #
        if "shape" in state:
            state["_shape"] = state.get("shape")
            del state["shape"]
        if "np_random" in state:
            state["_np_random"] = state["np_random"]
            del state["np_random"]

        # Update our state
        self.__dict__.update(state)

    def to_jsonable(self, sample_n: Sequence[T_cov]) -> list[Any]:
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return list(sample_n)

    def from_jsonable(self, sample_n: list[Any]) -> list[T_cov]:
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n
