---
title: Functional
---

# Functional Env

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

## Converting Jax-based Functional environments to standard Env

```{eval-rst}
.. autoclass:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv

    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv.reset
    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv.step
    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv.render
```
