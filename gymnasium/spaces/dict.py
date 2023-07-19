"""Implementation of a space that represents the cartesian product of other spaces as a dictionary."""
from __future__ import annotations

import collections.abc
import typing
from collections import OrderedDict
from typing import Any, KeysView, Sequence

import numpy as np

from gymnasium.spaces.space import Space


class Dict(Space[typing.Dict[str, Any]], typing.Mapping[str, Space[Any]]):
    """A dictionary of :class:`Space` instances.

    Elements of this space are (ordered) dictionaries of elements from the constituent spaces.

    Example:
        >>> from gymnasium.spaces import Dict, Box, Discrete
        >>> observation_space = Dict({"position": Box(-1, 1, shape=(2,)), "color": Discrete(3)}, seed=42)
        >>> observation_space.sample()
        OrderedDict([('color', 0), ('position', array([-0.3991573 ,  0.21649833], dtype=float32))])

        With a nested dict:

        >>> from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
        >>> Dict(  # doctest: +SKIP
        ...     {
        ...         "ext_controller": MultiDiscrete([5, 2, 2]),
        ...         "inner_state": Dict(
        ...             {
        ...                 "charge": Discrete(100),
        ...                 "system_checks": MultiBinary(10),
        ...                 "job_status": Dict(
        ...                     {
        ...                         "task": Discrete(5),
        ...                         "progress": Box(low=0, high=100, shape=()),
        ...                     }
        ...                 ),
        ...             }
        ...         ),
        ...     }
        ... )

    It can be convenient to use :class:`Dict` spaces if you want to make complex observations or actions more human-readable.
    Usually, it will not be possible to use elements of this space directly in learning code. However, you can easily
    convert `Dict` observations to flat arrays by using a :class:`gymnasium.wrappers.FlattenObservation` wrapper.
    Similar wrappers can be implemented to deal with :class:`Dict` actions.
    """

    def __init__(
        self,
        spaces: None | dict[str, Space] | Sequence[tuple[str, Space]] = None,
        seed: dict | int | np.random.Generator | None = None,
        **spaces_kwargs: Space,
    ):
        """Constructor of :class:`Dict` space.

        This space can be instantiated in one of two ways: Either you pass a dictionary
        of spaces to :meth:`__init__` via the ``spaces`` argument, or you pass the spaces as separate
        keyword arguments (where you will need to avoid the keys ``spaces`` and ``seed``)

        Args:
            spaces: A dictionary of spaces. This specifies the structure of the :class:`Dict` space
            seed: Optionally, you can use this argument to seed the RNGs of the spaces that make up the :class:`Dict` space.
            **spaces_kwargs: If ``spaces`` is ``None``, you need to pass the constituent spaces as keyword arguments, as described above.
        """
        # Convert the spaces into an OrderedDict
        if isinstance(spaces, collections.abc.Mapping) and not isinstance(
            spaces, OrderedDict
        ):
            try:
                spaces = OrderedDict(sorted(spaces.items()))
            except TypeError:
                # Incomparable types (e.g. `int` vs. `str`, or user-defined types) found.
                # The keys remain in the insertion order.
                spaces = OrderedDict(spaces.items())
        elif isinstance(spaces, Sequence):
            spaces = OrderedDict(spaces)
        elif spaces is None:
            spaces = OrderedDict()
        else:
            assert isinstance(
                spaces, OrderedDict
            ), f"Unexpected Dict space input, expecting dict, OrderedDict or Sequence, actual type: {type(spaces)}"

        # Add kwargs to spaces to allow both dictionary and keywords to be used
        for key, space in spaces_kwargs.items():
            if key not in spaces:
                spaces[key] = space
            else:
                raise ValueError(
                    f"Dict space keyword '{key}' already exists in the spaces dictionary."
                )

        self.spaces: dict[str, Space[Any]] = spaces
        for key, space in self.spaces.items():
            assert isinstance(
                space, Space
            ), f"Dict space element is not an instance of Space: key='{key}', space={space}"

        # None for shape and dtype, since it'll require special handling
        super().__init__(None, None, seed)  # type: ignore

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return all(space.is_np_flattenable for space in self.spaces.values())

    def seed(self, seed: dict[str, Any] | int | None = None) -> list[int]:
        """Seed the PRNG of this space and all subspaces.

        Depending on the type of seed, the subspaces will be seeded differently

        * ``None`` - All the subspaces will use a random initial seed
        * ``Int`` - The integer is used to seed the :class:`Dict` space that is used to generate seed values for each of the subspaces. Warning, this does not guarantee unique seeds for all of the subspaces.
        * ``Dict`` - Using all the keys in the seed dictionary, the values are used to seed the subspaces. This allows the seeding of multiple composite subspaces (``Dict["space": Dict[...], ...]`` with ``{"space": {...}, ...}``).

        Args:
            seed: An optional list of ints or int to seed the (sub-)spaces.
        """
        seeds: list[int] = []

        if isinstance(seed, dict):
            assert (
                seed.keys() == self.spaces.keys()
            ), f"The seed keys: {seed.keys()} are not identical to space keys: {self.spaces.keys()}"
            for key in seed.keys():
                seeds += self.spaces[key].seed(seed[key])
        elif isinstance(seed, int):
            seeds = super().seed(seed)
            # Using `np.int32` will mean that the same key occurring is extremely low, even for large subspaces
            subseeds = self.np_random.integers(
                np.iinfo(np.int32).max, size=len(self.spaces)
            )
            for subspace, subseed in zip(self.spaces.values(), subseeds):
                seeds += subspace.seed(int(subseed))
        elif seed is None:
            for space in self.spaces.values():
                seeds += space.seed(None)
        else:
            raise TypeError(
                f"Expected seed type: dict, int or None, actual type: {type(seed)}"
            )

        return seeds

    def sample(self, mask: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generates a single random sample from this space.

        The sample is an ordered dictionary of independent samples from the constituent spaces.

        Args:
            mask: An optional mask for each of the subspaces, expects the same keys as the space

        Returns:
            A dictionary with the same key and sampled values from :attr:`self.spaces`
        """
        if mask is not None:
            assert isinstance(
                mask, dict
            ), f"Expects mask to be a dict, actual type: {type(mask)}"
            assert (
                mask.keys() == self.spaces.keys()
            ), f"Expect mask keys to be same as space keys, mask keys: {mask.keys()}, space keys: {self.spaces.keys()}"
            return OrderedDict(
                [(k, space.sample(mask[k])) for k, space in self.spaces.items()]
            )

        return OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, dict) and x.keys() == self.spaces.keys():
            return all(x[key] in self.spaces[key] for key in self.spaces.keys())
        return False

    def __getitem__(self, key: str) -> Space[Any]:
        """Get the space that is associated to `key`."""
        return self.spaces[key]

    def keys(self) -> KeysView:
        """Returns the keys of the Dict."""
        return KeysView(self.spaces)

    def __setitem__(self, key: str, value: Space[Any]):
        """Set the space that is associated to `key`."""
        assert isinstance(
            value, Space
        ), f"Trying to set {key} to Dict space with value that is not a gymnasium space, actual type: {type(value)}"
        self.spaces[key] = value

    def __iter__(self):
        """Iterator through the keys of the subspaces."""
        yield from self.spaces

    def __len__(self) -> int:
        """Gives the number of simpler spaces that make up the `Dict` space."""
        return len(self.spaces)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return (
            "Dict(" + ", ".join([f"{k!r}: {s}" for k, s in self.spaces.items()]) + ")"
        )

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return (
            isinstance(other, Dict)
            # Comparison of `OrderedDict`s is order-sensitive
            and self.spaces == other.spaces  # OrderedDict.__eq__
        )

    def to_jsonable(self, sample_n: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        # serialize as dict-repr of vectors
        return {
            key: space.to_jsonable([sample[key] for sample in sample_n])
            for key, space in self.spaces.items()
        }

    def from_jsonable(
        self, sample_n: dict[str, list[Any]]
    ) -> list[OrderedDict[str, Any]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        dict_of_list: dict[str, list[Any]] = {
            key: space.from_jsonable(sample_n[key])
            for key, space in self.spaces.items()
        }

        n_elements = len(next(iter(dict_of_list.values())))
        result = [
            OrderedDict({key: value[n] for key, value in dict_of_list.items()})
            for n in range(n_elements)
        ]
        return result
