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

* :py:class:`Box` - Supports continuous (and discrete) vectors or matrices, used for vector observations, images, etc
* :py:class:`Discrete` - Supports a single discrete number of values with an optional start for the values
* :py:class:`MultiDiscrete` - Supports single or matrices of binary values, used for holding down a button or if an agent has an object
* :py:class:`MultiBinary` - Supports multiple discrete values with multiple axes, used for controller actions
* :py:class:`Text` - Supports strings, used for passing agent messages, mission details, etc 
```

## Composite Spaces

```{eval-rst}
* :py:class:`Dict` - Supports a dictionary of keys and subspaces, used for a fixed number of unordered spaces
* :py:class:`Tuple` - Supports a tuple of subspaces, used for multiple for a fixed number of ordered spaces
* :py:class:`Sequence` - Supports a variable number of instances of a single subspace, used for entities spaces or selecting a variable number of actions
* :py:class:`Graph` - Supports graph based actions or observations with discrete or continuous nodes and edge values.
```

## Utils

```{eval-rst}
* :py:class:`utils.flatdim` - The number of dimensions the flattened space will contain
* :py:class:`utils.flatten_space` - Flattens a space for which the `flattened` space instances will contain
* :py:class:`utils.flatten` - Flattens an instance of a space that is contained within the flattened version of the space
* :py:class:`utils.unflatten` - The reverse of the `flatten_space` function
```
