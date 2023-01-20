---
title: Spaces
---

# Spaces

```{toctree}
:hidden:
spaces/fundamental
spaces/composite
spaces/utils
spaces/vector_utils
```

```{eval-rst}
.. automodule:: gymnasium.spaces
```

## The Base Class

```{eval-rst}
.. autoclass:: gymnasium.spaces.Space
```

### Attributes

```{eval-rst}
.. autoproperty:: gymnasium.spaces.space.Space.shape
.. property:: Space.dtype

    Return the data type of this space.
.. autoproperty:: gymnasium.spaces.space.Space.is_np_flattenable
```

### Methods

Each space implements the following functions:

```{eval-rst}
.. autofunction:: gymnasium.spaces.space.Space.sample
.. autofunction:: gymnasium.spaces.space.Space.contains
.. autofunction:: gymnasium.spaces.space.Space.seed
.. autofunction:: gymnasium.spaces.space.Space.to_jsonable
.. autofunction:: gymnasium.spaces.space.Space.from_jsonable
```

## Fundamental Spaces

Gymnasium has a number of fundamental spaces that are used as building boxes for more complex spaces.

```{eval-rst}
.. currentmodule:: gymnasium.spaces

* :py:class:`Box` - Supports continuous (and discrete) vectors or matrices, used for vector observations, images, etc
* :py:class:`Discrete` - Supports a single discrete number of values with an optional start for the values
* :py:class:`MultiBinary` - Supports single or matrices of binary values, used for holding down a button or if an agent has an object
* :py:class:`MultiDiscrete` - Supports multiple discrete values with multiple axes, used for controller actions
* :py:class:`Text` - Supports strings, used for passing agent messages, mission details, etc
```

## Composite Spaces

Often environment spaces require joining fundamental spaces together for vectorised environments, separate agents or readability of the space.

```{eval-rst}
* :py:class:`Dict` - Supports a dictionary of keys and subspaces, used for a fixed number of unordered spaces
* :py:class:`Tuple` - Supports a tuple of subspaces, used for multiple for a fixed number of ordered spaces
* :py:class:`Sequence` - Supports a variable number of instances of a single subspace, used for entities spaces or selecting a variable number of actions
* :py:class:`Graph` - Supports graph based actions or observations with discrete or continuous nodes and edge values.
```

## Utils

Gymnasium contains a number of helpful utility functions for flattening and unflattening spaces.
This can be important for passing information to neural networks.

```{eval-rst}
* :py:class:`utils.flatdim` - The number of dimensions the flattened space will contain
* :py:class:`utils.flatten_space` - Flattens a space for which the `flattened` space instances will contain
* :py:class:`utils.flatten` - Flattens an instance of a space that is contained within the flattened version of the space
* :py:class:`utils.unflatten` - The reverse of the `flatten_space` function
```

## Vector Utils

When vectorizing environments, it is necessary to modify the observation and action spaces for new batched spaces sizes.
Therefore, Gymnasium provides a number of additional functions used when using a space with a Vector environment.

```{eval-rst}
.. currentmodule:: gymnasium

* :py:class:`vector.utils.batch_space`
* :py:class:`vector.utils.concatenate`
* :py:class:`vector.utils.iterate`
* :py:class:`vector.utils.create_empty_array`
* :py:class:`vector.utils.create_shared_memory`
* :py:class:`vector.utils.read_from_shared_memory`
* :py:class:`vector.utils.write_to_shared_memory`
```
