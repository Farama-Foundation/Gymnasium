# Core

## gymnasium.Env
```{eval-rst}
.. autoclass:: gymnasium.Env
.. autofunction:: gymnasium.Env.step
.. autofunction:: gymnasium.Env.reset
.. autofunction:: gymnasium.Env.render
```

### Attributes

```{eval-rst}
.. autoattribute:: gymnasium.Env.action_space

    The Space object corresponding to valid actions, all valid actions should be contained with the space. For example, if the action space is of type `Discrete` and gives the value `Discrete(2)`, this means there are two valid discrete actions: 0 & 1.
    
    .. code::

        >>> env.action_space
        Discrete(2)
        >>> env.observation_space
        Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)
```

```{eval-rst}
.. autoattribute:: gymnasium.Env.observation_space
    
    The Space object corresponding to valid observations, all valid observations should be contained with the space. For example, if the observation space is of type :class:`Box` and the shape of the object is ``(4,)``, this denotes a valid observation will be an array of 4 numbers. We can check the box bounds as well with attributes.
    
    .. code::

        >>> env.observation_space.high
        array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=float32)
        >>> env.observation_space.low
        array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38], dtype=float32)
``` 

```{eval-rst}
.. autoattribute:: gymnasium.Env.reward_range
    
    A tuple corresponding to the minimum and maximum possible rewards for an agent over an episode. The default reward range is set to :math:`(-\infty,+\infty)`.
``` 

### Additional Methods

```{eval-rst}
.. autofunction:: gymnasium.Env.close
```

## gymnasium.Wrapper

```{eval-rst}
.. autoclass:: gymnasium.Wrapper
```

### Attributes

```{eval-rst}
.. autofunction:: gymnasium.Wrapper.action_space
.. autofunction:: gymnasium.Wrapper.observation_space
.. autofunction:: gymnasium.Wrapper.reward_range
.. autofunction:: gymnasium.Wrapper.spec
.. autofunction:: gymnasium.Wrapper.metadata
.. autofunction:: gymnasium.Wrapper.np_random
.. autofunction:: gymnasium.Wrapper.unwrapped
```
