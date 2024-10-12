"""Allows attributes passed to `RecordConstructorArgs` to be saved. This is used by the `Wrapper.spec` to know the constructor arguments of implemented wrappers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class RecordConstructorArgs:
    """Records all arguments passed to constructor to `_saved_kwargs`.

    This can be used to save and reproduce class constructor arguments.

    Note:
        If two class inherit from RecordConstructorArgs then the first class to call `RecordConstructorArgs.__init__(self, ...)` will have
        their kwargs saved will all subsequent `RecordConstructorArgs.__init__` being ignored.

        Therefore, always call `RecordConstructorArgs.__init__` before the `Class.__init__`


    """

    def __init__(self, *, _disable_deepcopy: bool = False, **kwargs: Any):
        """Records all arguments passed to constructor to `_saved_kwargs`.

        Args:
            _disable_deepcopy: If to not deepcopy the kwargs passed
            **kwargs: Arguments to save
        """
        # See class docstring for explanation
        if not hasattr(self, "_saved_kwargs"):
            if _disable_deepcopy is False:
                kwargs = deepcopy(kwargs)
            self._saved_kwargs: dict[str, Any] = kwargs
