# Spaces

```{toctree}
:hidden:
spaces/fundamental
spaces/composite
spaces/utils
```


```{eval-rst}
.. autoclass:: gymnasium.spaces.Space
```

## General Functions

Each space implements the following functions:

```{eval-rst}
.. autofunction:: gymnasium.spaces.Space.sample

.. autofunction:: gymnasium.spaces.Space.contains

.. autoproperty:: gymnasium.spaces.Space.shape

.. property:: gymnasium.spaces.Space.dtype

    Return the data type of this space.

.. autofunction:: gymnasium.spaces.Space.seed

.. autofunction:: gymnasium.spaces.Space.to_jsonable

.. autofunction:: gymnasium.spaces.Space.from_jsonable
``` 

## Fundamental Spaces

```{eval-rst}
.. currentmodule:: gymnasium.spaces

* :py:class:`Box`
* :py:class:`Discrete`
* :py:class:`MultiBinary`
* :py:class:`MultiDiscrete`
* :py:class:`Text`
```

## Composite

```{eval-rst}
* :py:class:`Dict`
* :py:class:`Tuple`
* :py:class:`Sequence`
* :py:class:`Graph`
```

## Utils

```{eval-rst}
* :py:class:`utils.flatdim`
* :py:class:`utils.flatten_space`
* :py:class:`utils.flatten`
* :py:class:`utils.unflatten`
```
