# Core

## gymnasium.Env

```{eval-rst}
.. autofunction:: gymnasium.Env.step
```

```{eval-rst}
.. autofunction:: gymnasium.Env.reset
```

```{eval-rst}
.. autofunction:: gymnasium.Env.render
```

### Attributes

```{eval-rst}
.. autoattribute:: gymnasium.Env.action_space

    This attribute gives the format of valid actions. It is of datatype `Space` provided by Gymnasium. For example, if the action space is of type `Discrete` and gives the value `Discrete(2)`, this means there are two valid discrete actions: 0 & 1.

    .. code::
    
        >>> env.action_space
        Discrete(2)
        >>> env.observation_space
        Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)
```

```{eval-rst}
.. autoattribute:: gymnasium.Env.observation_space

    This attribute gives the format of valid observations. It is of datatype :class:`Space` provided by Gymnasium. For example, if the observation space is of type :class:`Box` and the shape of the object is ``(4,)``, this denotes a valid observation will be an array of 4 numbers. We can check the box bounds as well with attributes.

    .. code::

        >>> env.observation_space.high
        array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=float32)
        >>> env.observation_space.low
        array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38], dtype=float32)
``` 

```{eval-rst}
.. autoattribute:: gymnasium.Env.reward_range

    This attribute is a tuple corresponding to min and max possible rewards. Default range is set to ``(-inf,+inf)``. You can set it if you want a narrower range.
``` 

### Additional Methods

```{eval-rst}
.. autofunction:: gymnasium.Env.close
``` 
 
```{eval-rst}
.. autofunction:: gymnasium.Env.seed
```


## gymnasium.Wrapper

```{eval-rst}
.. autoclass:: gymnasium.Wrapper
```

## gymnasium.ObservationWrapper

```{eval-rst}
.. autoclass:: gymnasium.ObservationWrapper
```


## gymnasium.RewardWrapper

```{eval-rst}
.. autoclass:: gymnasium.RewardWrapper
```

## gymnasium.ActionWrapper

```{eval-rst}
.. autoclass:: gymnasium.ActionWrapper
```