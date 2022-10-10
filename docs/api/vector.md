---
title: Vector
---

# Vector

## Gymnasium.vector.VectorEnv

```{eval-rst}
.. autoclass:: gymnasium.vector.VectorEnv
```

### Methods

```{eval-rst}
.. automethod:: gymnasium.vector.VectorEnv.reset

.. automethod:: gymnasium.vector.VectorEnv.step
```

### Attributes

```{eval-rst}
.. attribute:: gymnasium.vector.VectorEnv.action_space

    The (batched) action space. The input actions of `step` must be valid elements of `action_space`.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.action_space
        MultiDiscrete([2 2 2])

.. attribute:: gymnasium.vector.VectorEnv.observation_space

    The (batched) observation space. The observations returned by `reset` and `step` are valid elements of `observation_space`.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.observation_space
        Box([[-4.8 ...]], [[4.8 ...]], (3, 4), float32)

.. attribute:: gymnasium.vector.VectorEnv.single_action_space

    The action space of an environment copy.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.single_action_space
        Discrete(2)

.. attribute:: gymnasium.vector.VectorEnv.single_observation_space

    The observation space of an environment copy.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.single_action_space
        Box([-4.8 ...], [4.8 ...], (4,), float32)
```

## Making Vector Environments

```{eval-rst}
.. autofunction:: gymnasium.vector.make
```