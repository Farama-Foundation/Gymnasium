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

    .. automethod:: __init__
    .. automethod:: is_bounded
    .. automethod:: sample
``` 

### Discrete

```{eval-rst}
.. autoclass:: gymnasium.spaces.Discrete
 
    .. autoclass:: __init__
    .. automethod:: sample
``` 

### MultiBinary

```{eval-rst}
.. autoclass:: gymnasium.spaces.MultiBinary

    .. automethod:: __init__
    .. automethod:: sample
``` 

### MultiDiscrete

```{eval-rst}
.. autoclass:: gymnasium.spaces.MultiDiscrete

    .. automethod:: __init__
    .. automethod:: sample
``` 

### Text

```{eval-rst}
.. autoclass:: gymnasium.spaces.Text

    .. automethod:: __init__
    .. automethod:: sample
``` 

## Composite Spaces

### Dict

```{eval-rst}
.. autoclass:: gymnasium.spaces.Dict

    .. automethod:: __init__
    .. automethod:: sample
``` 

### Graph

```{eval-rst}
.. autoclass:: gymnasium.spaces.Graph

    .. automethod:: __init__
    .. automethod:: sample
```

### Sequence

```{eval-rst}
.. autoclass:: gymnasium.spaces.Sequence

    .. automethod:: __init__
    .. automethod:: sample
``` 

### Tuple

```{eval-rst}
.. autoclass:: gymnasium.spaces.Tuple

    .. automethod:: __init__
    .. automethod:: sample
``` 
