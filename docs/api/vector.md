---
title: Vector
---

# Vectorize

```{toctree}
:hidden:
vector/wrappers
vector/async_vector_env
vector/sync_vector_env
vector/utils
```

## Gymnasium.vector.VectorEnv

```{eval-rst}
.. autoclass:: gymnasium.vector.VectorEnv
```

### Methods
```{eval-rst}
.. automethod:: gymnasium.vector.VectorEnv.step
.. automethod:: gymnasium.vector.VectorEnv.reset
.. automethod:: gymnasium.vector.VectorEnv.render
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

    The observation space of a sub-environment.

.. autoattribute:: gymnasium.vector.VectorEnv.spec

    The ``EnvSpec`` of the environment normally set during :py:meth:`gymnasium.make_vec`

.. autoattribute:: gymnasium.vector.VectorEnv.metadata

    The metadata of the environment containing rendering modes, rendering fps, etc

.. autoattribute:: gymnasium.vector.VectorEnv.render_mode

    The render mode of the environment which should follow similar specifications to `Env.render_mode`.

.. autoattribute:: gymnasium.vector.VectorEnv.closed

    If the vector environment has been closed already.
```

### Additional Methods

```{eval-rst}
.. autoproperty:: gymnasium.vector.VectorEnv.unwrapped
.. autoproperty:: gymnasium.vector.VectorEnv.np_random
.. autoproperty:: gymnasium.vector.VectorEnv.np_random_seed
```

## Making Vector Environments

```{eval-rst}
To create vector environments, gymnasium provides :func:`gymnasium.make_vec` as an equivalent function to :func:`gymnasium.make`.
```
