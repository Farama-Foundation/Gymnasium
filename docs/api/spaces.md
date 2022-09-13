# Spaces

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

### Box

```{eval-rst}
.. autoclass:: gymnasium.spaces.Box

    .. automethod:: is_bounded
    .. automethod:: sample
``` 

### Discrete

```{eval-rst}
.. autoclass:: gymnasium.spaces.Discrete
 
    .. automethod:: sample
``` 

### MultiBinary

```{eval-rst}
.. autoclass:: gymnasium.spaces.MultiBinary

    .. automethod:: sample
``` 

### MultiDiscrete

```{eval-rst}
.. autoclass:: gymnasium.spaces.MultiDiscrete

    .. automethod:: sample
``` 

### Text

```{eval-rst}
.. autoclass:: gymnasium.spaces.Text

    .. automethod:: sample
``` 

## Composite Spaces

### Dict

```{eval-rst}
.. autoclass:: gymnasium.spaces.Dict

    .. automethod:: sample
``` 

### Graph

```{eval-rst}
.. autoclass:: gymnasium.spaces.Graph

    .. automethod:: sample
```

### Sequence

```{eval-rst}
.. autoclass:: gymnasium.spaces.Sequence

    .. automethod:: sample
``` 

### Tuple

```{eval-rst}
.. autoclass:: gymnasium.spaces.Tuple

    .. automethod:: sample
``` 
