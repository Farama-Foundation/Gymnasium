---
title: Functional
---

# Functional Env

```{warning}
The functional API is experimental, in `gymnasium.experimental`, and may change in future releases.
```

```{eval-rst}
.. autoclass:: gymnasium.experimental.functional.FuncEnv

    .. automethod:: gymnasium.experimental.functional.FuncEnv.transform

    .. automethod:: gymnasium.experimental.functional.FuncEnv.initial

    .. automethod:: gymnasium.experimental.functional.FuncEnv.transition
    .. automethod:: gymnasium.experimental.functional.FuncEnv.observation
    .. automethod:: gymnasium.experimental.functional.FuncEnv.reward
    .. automethod:: gymnasium.experimental.functional.FuncEnv.terminal

    .. automethod:: gymnasium.experimental.functional.FuncEnv.state_info
    .. automethod:: gymnasium.experimental.functional.FuncEnv.transition_info

    .. automethod:: gymnasium.experimental.functional.FuncEnv.render_init
    .. automethod:: gymnasium.experimental.functional.FuncEnv.render_image
    .. automethod:: gymnasium.experimental.functional.FuncEnv.render_close
```

## Converting Jax-based functional environments to standard Env and VectorEnv

```{eval-rst}
.. autoclass:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv

    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv.reset
    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv.step
    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv.render

.. autoclass:: gymnasium.envs.functional_jax_env.FunctionalJaxVectorEnv

    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxVectorEnv.reset
    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxVectorEnv.step
    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxVectorEnv.render
```
