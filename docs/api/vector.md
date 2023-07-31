---
title: Vector
---

# Vector

```{toctree}
:hidden:
vector/async_vector_env
vector/sync_vector_env
vector/utils
vector/wrappers
```

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
.. autoattribute:: gymnasium.vector.VectorEnv.num_envs

    The number of sub-environments in the vector environment.

.. autoattribute:: gymnasium.vector.VectorEnv.action_space

    The (batched) action space. The input actions of `step` must be valid elements of `action_space`.

.. autoattribute:: gymnasium.vector.VectorEnv.observation_space

    The (batched) observation space. The observations returned by `reset` and `step` are valid elements of `observation_space`.

.. autoattribute:: gymnasium.vector.VectorEnv.single_action_space

    The action space of a sub-environment.

.. autoattribute:: gymnasium.vector.VectorEnv.single_observation_space

    The observation space of an environment copy.

.. autoattribute:: gymnasium.vector.VectorEnv.spec

    The ``EnvSpec`` of the environment normally set during :py:meth:`gymnasium.make_vec`
```

### Additional Methods

```{eval-rst}
.. autoproperty:: gymnasium.vector.VectorEnv.unwrapped
.. autoproperty:: gymnasium.vector.VectorEnv.np_random
```

## Making Vector Environments

```{eval-rst}
.. autofunction:: gymnasium.make_vec
```
