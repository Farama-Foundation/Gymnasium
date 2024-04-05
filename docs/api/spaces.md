---
title: Spaces
---

# Spaces

```{toctree}
:hidden:
spaces/fundamental
spaces/composite
spaces/utils
```

```{eval-rst}
.. automodule:: gymnasium.spaces

.. autoclass:: gymnasium.spaces.Space
```

## Attributes
```{eval-rst}
.. py:currentmodule:: gymnasium.spaces

.. autoproperty:: Space.shape
.. property:: Space.dtype

    Return the data type of this space.
.. autoproperty:: Space.is_np_flattenable
.. autoproperty:: Space.np_random
```

## Methods
Each space implements the following functions:

```{eval-rst}
.. py:currentmodule:: gymnasium.spaces

.. automethod:: Space.sample
.. automethod:: Space.contains
.. automethod:: Space.seed
.. automethod:: Space.to_jsonable
.. automethod:: Space.from_jsonable
```

## Fundamental Spaces

Gymnasium has a number of fundamental spaces that are used as building boxes for more complex spaces.

```{eval-rst}
.. py:currentmodule:: gymnasium.spaces

* :class:`Box` - Supports continuous (and discrete) vectors or matrices, used for vector observations, images, etc
* :class:`Discrete` - Supports a single discrete number of values with an optional start for the values
* :class:`MultiBinary` - Supports single or matrices of binary values, used for holding down a button or if an agent has an object
* :class:`MultiDiscrete` - Supports multiple discrete values with multiple axes, used for controller actions
* :class:`Text` - Supports strings, used for passing agent messages, mission details, etc
```

## Composite Spaces

Often environment spaces require joining fundamental spaces together for vectorised environments, separate agents or readability of the space.

```{eval-rst}
.. py:currentmodule:: gymnasium.spaces

* :class:`Dict` - Supports a dictionary of keys and subspaces, used for a fixed number of unordered spaces
* :class:`Tuple` - Supports a tuple of subspaces, used for multiple for a fixed number of ordered spaces
* :class:`Sequence` - Supports a variable number of instances of a single subspace, used for entities spaces or selecting a variable number of actions
* :class:`Graph` - Supports graph based actions or observations with discrete or continuous nodes and edge values
* :class:`OneOf` - Supports optional action spaces such that an action can be one of N possible subspaces
```

## Utility functions

Gymnasium contains a number of helpful utility functions for flattening and unflattening spaces.
This can be important for passing information to neural networks.

```{eval-rst}
.. py:currentmodule:: gymnasium.spaces

* :class:`utils.flatdim` - The number of dimensions the flattened space will contain
* :class:`utils.flatten_space` - Flattens a space for which the :class:`utils.flattened` space instances will contain
* :class:`utils.flatten` - Flattens an instance of a space that is contained within the flattened version of the space
* :class:`utils.unflatten` - The reverse of the :class:`utils.flatten_space` function
```

## Vector Utility functions

When vectorizing environments, it is necessary to modify the observation and action spaces for new batched spaces sizes.
Therefore, Gymnasium provides a number of additional functions used when using a space with a Vector environment.

```{eval-rst}
.. py:currentmodule:: gymnasium

* :class:`vector.utils.batch_space` - Transforms a space into the equivalent space for ``n`` users
* :class:`vector.utils.concatenate` - Concatenates a space's samples into a pre-generated space
* :class:`vector.utils.iterate` - Iterate over the batched space's samples
* :class:`vector.utils.create_empty_array` - Creates an empty sample for an space (generally used with ``concatenate``)
* :class:`vector.utils.create_shared_memory` - Creates a shared memory for asynchronous (multiprocessing) environment
* :class:`vector.utils.read_from_shared_memory` - Reads a shared memory for asynchronous (multiprocessing) environment
* :class:`vector.utils.write_to_shared_memory` - Write to a shared memory for asynchronous (multiprocessing) environment
```
