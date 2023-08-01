"""Decorator for pickling and unpickling objects via their constructor arguments."""
from __future__ import annotations

from typing import Any


def ezpickle(func):
    """Decorator for pickling and unpickling objects via their constructor arguments.

    Example:
        >>> class Animal:
        ...     pass

        >>> class Dog(Animal):
        ...    @ezpickle
        ...    def __init__(self, furcolor, tailkind="bushy"):
        ...        super.__init__()
        ...

    When this object is unpickled, a new ``Dog`` will be constructed by passing
    the provided furcolor and tailkind into the constructor.
    However, philosophers are still not sure whether it is still the same dog.

    This is generally needed only for environments which wrap C/C++ code, such
    as MuJoCo and Atari.
    """

    def __getstate__(self):
        """Returns the object pickle state with args and kwargs."""
        return {
            "_ezpickle_args": self._ezpickle_args,
            "_ezpickle_kwargs": self._ezpickle_kwargs,
        }

    def __setstate__(self, d):
        """Sets the object pickle state using d."""
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)

    def wrapper(self, *args: tuple[Any], **kwargs: dict[str, Any]):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs
        self.__class__.__getstate__ = __getstate__
        self.__class__.__setstate__ = __setstate__
        return func(self, *args, **kwargs)

    return wrapper
