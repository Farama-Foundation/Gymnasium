---
title: Env
---

# Env

```{eval-rst}
.. autoclass:: gymnasium.Env
```

## Methods
```{eval-rst}
.. automethod:: gymnasium.Env.step
.. automethod:: gymnasium.Env.reset
.. automethod:: gymnasium.Env.render
.. automethod:: gymnasium.Env.close
```

## Attributes
```{eval-rst}
.. autoattribute:: gymnasium.Env.action_space

    The Space object corresponding to valid actions, all valid actions should be contained with the space. For example, if the action space is of type `Discrete` and gives the value `Discrete(2)`, this means there are two valid discrete actions: `0` & `1`.

    .. code::

        >>> env.action_space
        Discrete(2)
        >>> env.observation_space
        Box(-inf, inf, (4,), float32)

.. autoattribute:: gymnasium.Env.observation_space

    The Space object corresponding to valid observations, all valid observations should be contained with the space. For example, if the observation space is of type :class:`Box` and the shape of the object is ``(4,)``, this denotes a valid observation will be an array of 4 numbers. We can check the box bounds as well with attributes.

    .. code::

        >>> env.observation_space.high
        array([4.8000002e+00, inf, 4.1887903e-01, inf], dtype=float32)
        >>> env.observation_space.low
        array([-4.8000002e+00, -inf, -4.1887903e-01, -inf], dtype=float32)

.. autoattribute:: gymnasium.Env.metadata

    The metadata of the environment containing rendering modes, rendering fps, etc

.. autoattribute:: gymnasium.Env.render_mode

    The render mode of the environment determined at initialisation

.. autoattribute:: gymnasium.Env.spec

    The :class:`EnvSpec` of the environment normally set during :py:meth:`gymnasium.make`

.. autoproperty:: gymnasium.Env.unwrapped
.. autoproperty:: gymnasium.Env.np_random
.. autoproperty:: gymnasium.Env.np_random_seed
```

## Implementing environments

```{eval-rst}
.. py:currentmodule:: gymnasium

When implementing an environment, the :meth:`Env.reset` and :meth:`Env.step` functions must be created to describe the dynamics of the environment. For more information, see the environment creation tutorial.
```

## Creating environments

```{eval-rst}
.. py:currentmodule:: gymnasium

To create an environment, gymnasium provides :meth:`make` to initialise the environment along with several important wrappers. Furthermore, gymnasium provides :meth:`make_vec` for creating vector environments and to view all the environment that can be created use :meth:`pprint_registry`.
```
