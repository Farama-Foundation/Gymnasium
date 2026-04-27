"""Class for pickling and unpickling objects via their constructor arguments."""

from typing import Any


class EzPickle:
    """Objects that are pickled and unpickled via their constructor arguments.

    Example:
        >>> class Animal: pass
        >>> class Dog(Animal, EzPickle):
        ...    def __init__(self, furcolor, tailkind="bushy"):
        ...        Animal.__init__(self)
        ...        EzPickle.__init__(self, furcolor, tailkind)

    When this object is unpickled, a new ``Dog`` will be constructed by passing the provided furcolor and tailkind into the constructor.
    However, philosophers are still not sure whether it is still the same dog.

    This is generally needed only for environments which wrap C/C++ code, such as MuJoCo and Atari.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Uses the ``args`` and ``kwargs`` from the object's constructor for pickling."""
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self) -> dict[str, Any]:
        """Returns the object pickle state with args and kwargs."""
        return {
            "_ezpickle_args": self._ezpickle_args,
            "_ezpickle_kwargs": self._ezpickle_kwargs,
        }

    def __setstate__(self, d: dict[str, Any]) -> None:
        """Sets the object pickle state using d."""
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)
