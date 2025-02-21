"""Implementation of a space that represents textual strings."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from gymnasium.spaces.space import Space


alphanumeric: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


class Text(Space[str]):
    r"""A space representing a string comprised of characters from a given charset.

    Example:
        >>> from gymnasium.spaces import Text
        >>> # {"", "B5", "hello", ...}
        >>> Text(5)
        Text(1, 5, charset=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz)
        >>> # {"0", "42", "0123456789", ...}
        >>> import string
        >>> Text(min_length = 1,
        ...      max_length = 10,
        ...      charset = string.digits)
        Text(1, 10, charset=0123456789)
    """

    def __init__(
        self,
        max_length: int,
        *,
        min_length: int = 1,
        charset: frozenset[str] | str = alphanumeric,
        seed: int | np.random.Generator | None = None,
    ):
        r"""Constructor of :class:`Text` space.

        Both bounds for text length are inclusive.

        Args:
            min_length (int): Minimum text length (in characters). Defaults to 1 to prevent empty strings.
            max_length (int): Maximum text length (in characters).
            charset (Union[set], str): Character set, defaults to the lower and upper english alphabet plus latin digits.
            seed: The seed for sampling from the space.
        """
        assert np.issubdtype(
            type(min_length), np.integer
        ), f"Expects the min_length to be an integer, actual type: {type(min_length)}"
        assert np.issubdtype(
            type(max_length), np.integer
        ), f"Expects the max_length to be an integer, actual type: {type(max_length)}"
        assert (
            0 <= min_length
        ), f"Minimum text length must be non-negative, actual value: {min_length}"
        assert (
            min_length <= max_length
        ), f"The min_length must be less than or equal to the max_length, min_length: {min_length}, max_length: {max_length}"

        self.min_length: int = int(min_length)
        self.max_length: int = int(max_length)

        self._char_set: frozenset[str] = frozenset(charset)
        self._char_list: tuple[str, ...] = tuple(charset)
        self._char_index: dict[str, np.int32] = {
            val: np.int32(i) for i, val in enumerate(tuple(charset))
        }
        self._char_str: str = "".join(sorted(tuple(charset)))

        # As the shape is dynamic (between min_length and max_length) then None
        super().__init__(dtype=str, seed=seed)

    def sample(
        self,
        mask: None | (tuple[int | None, NDArray[np.int8] | None]) = None,
        probability: None | (tuple[int | None, NDArray[np.float64] | None]) = None,
    ) -> str:
        """Generates a single random sample from this space with by default a random length between ``min_length`` and ``max_length`` and sampled from the ``charset``.

        Args:
            mask: An optional tuples of length and mask for the text.
                The length is expected to be between the ``min_length`` and ``max_length``.
                Otherwise, a random integer between ``min_length`` and ``max_length`` is selected.
                For the mask, we expect a numpy array of length of the charset passed with ``dtype == np.int8``.
                If the charlist mask is all zero then an empty string is returned no matter the ``min_length``
            probability: An optional tuples of length and probability mask for the text.
                The length is expected to be between the ``min_length`` and ``max_length``.
                Otherwise, a random integer between ``min_length`` and ``max_length`` is selected.
                For the probability mask, we expect a numpy array of length of the charset passed with ``dtype == np.float64``.
                The sum of the probability mask should be 1, otherwise an exception is raised.

        Returns:
            A sampled string from the space
        """
        if mask is not None and probability is not None:
            raise ValueError(
                f"Only one of `mask` or `probability` can be provided, actual values: mask={mask}, probability={probability}"
            )
        elif mask is not None:
            length, charlist_mask = self._validate_mask(mask, np.int8, "mask")

            if charlist_mask is not None:
                assert np.all(
                    np.logical_or(charlist_mask == 0, charlist_mask == 1)
                ), f"Expects all mask values to 0 or 1, actual values: {charlist_mask}"

                # normalise the mask to use as a probability
                if np.sum(charlist_mask) > 0:
                    charlist_mask = charlist_mask / np.sum(charlist_mask)
        elif probability is not None:
            length, charlist_mask = self._validate_mask(
                probability, np.float64, "probability"
            )

            if charlist_mask is not None:
                assert np.all(
                    np.logical_and(charlist_mask >= 0, charlist_mask <= 1)
                ), f"Expects all probability mask values to be within 0 and 1, actual values: {charlist_mask}"
                assert np.isclose(
                    np.sum(charlist_mask), 1
                ), f"Expects the sum of the probability mask to be 1, actual sum: {np.sum(charlist_mask)}"
        else:
            length = charlist_mask = None

        if length is None:
            length = self.np_random.integers(self.min_length, self.max_length + 1)
        if charlist_mask is None:  # uniform sampling
            charlist_mask = np.ones(len(self.character_set)) / len(self.character_set)

        if np.all(charlist_mask == 0):
            if self.min_length == 0:
                return ""
            else:
                # Otherwise the string will not be contained in the space
                raise ValueError(
                    f"Trying to sample with a minimum length > 0 (actual minimum length={self.min_length}) but the character mask is all zero meaning that no character could be sampled."
                )

        string = self.np_random.choice(
            self.character_list, size=length, p=charlist_mask
        )
        return "".join(string)

    def _validate_mask(
        self,
        mask: tuple[int | None, NDArray[np.int8] | NDArray[np.float64] | None],
        expected_dtype: np.dtype,
        mask_type: str,
    ) -> tuple[int | None, NDArray[np.int8] | NDArray[np.float64] | None]:
        assert isinstance(
            mask, tuple
        ), f"Expects the `{mask_type}` type to be a tuple, actual type: {type(mask)}"
        assert (
            len(mask) == 2
        ), f"Expects the `{mask_type}` length to be two, actual length: {len(mask)}"
        length, charlist_mask = mask

        if length is not None:
            assert np.issubdtype(
                type(length), np.integer
            ), f"Expects the Text sample length to be an integer, actual type: {type(length)}"
            assert (
                self.min_length <= length <= self.max_length
            ), f"Expects the Text sample length be between {self.min_length} and {self.max_length}, actual length: {length}"
        if charlist_mask is not None:
            assert isinstance(
                charlist_mask, np.ndarray
            ), f"Expects the Text sample `{mask_type}` to be an np.ndarray, actual type: {type(charlist_mask)}"
            assert (
                charlist_mask.dtype == expected_dtype
            ), f"Expects the Text sample `{mask_type}` to be type {expected_dtype}, actual dtype: {charlist_mask.dtype}"
            assert charlist_mask.shape == (
                len(self.character_set),
            ), f"expects the Text sample `{mask_type}` to be {(len(self.character_set),)}, actual shape: {charlist_mask.shape}"

        return length, charlist_mask

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, str):
            if self.min_length <= len(x) <= self.max_length:
                return all(c in self.character_set for c in x)
        return False

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"Text({self.min_length}, {self.max_length}, charset={self.characters})"

    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Text)
            and self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.character_set == other.character_set
        )

    @property
    def character_set(self) -> frozenset[str]:
        """Returns the character set for the space."""
        return self._char_set

    @property
    def character_list(self) -> tuple[str, ...]:
        """Returns a tuple of characters in the space."""
        return self._char_list

    def character_index(self, char: str) -> np.int32:
        """Returns a unique index for each character in the space's character set."""
        return self._char_index[char]

    @property
    def characters(self) -> str:
        """Returns a string with all Text characters."""
        return self._char_str

    @property
    def is_np_flattenable(self) -> bool:
        """The flattened version is an integer array for each character, padded to the max character length."""
        return True
