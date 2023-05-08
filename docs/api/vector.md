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

.. automethod:: gymnasium.vector.VectorEnv.close
```

### Attributes

```{eval-rst}
.. attribute:: action_space

    The (batched) action space. The input actions of `step` must be valid elements of `action_space`.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.action_space
        MultiDiscrete([2 2 2])

.. attribute:: observation_space

    The (batched) observation space. The observations returned by `reset` and `step` are valid elements of `observation_space`.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.observation_space
        Box([[-4.8 ...]], [[4.8 ...]], (3, 4), float32)

.. attribute:: single_action_space

    The action space of an environment copy.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.single_action_space
        Discrete(2)

.. attribute:: single_observation_space

    The observation space of an environment copy.::

        >>> envs = gymnasium.vector.make("CartPole-v1", num_envs=3)
        >>> envs.single_observation_space
        Box([-4.8 ...], [4.8 ...], (4,), float32)
```

## Making Vector Environments

```{eval-rst}
.. autofunction:: gymnasium.vector.make
```

## Async Vector Env

```{eval-rst}
.. autoclass:: gymnasium.vector.AsyncVectorEnv
```

## Sync Vector Env

```{eval-rst}
.. autoclass:: gymnasium.vector.SyncVectorEnv
```
