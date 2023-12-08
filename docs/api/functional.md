---
title: Functional
---

# Functional Env

```{eval-rst}
.. autoclass:: gymnasium.functional.FuncEnv

    .. automethod:: gymnasium.functional.FuncEnv.transform

    .. automethod:: gymnasium.functional.FuncEnv.initial

    .. automethod:: gymnasium.functional.FuncEnv.transition
    .. automethod:: gymnasium.functional.FuncEnv.observation
    .. automethod:: gymnasium.functional.FuncEnv.reward
    .. automethod:: gymnasium.functional.FuncEnv.terminal

    .. automethod:: gymnasium.functional.FuncEnv.state_info
    .. automethod:: gymnasium.functional.FuncEnv.transition_info

    .. automethod:: gymnasium.functional.FuncEnv.render_init
    .. automethod:: gymnasium.functional.FuncEnv.render_image
    .. automethod:: gymnasium.functional.FuncEnv.render_close
```

## Converting Jax-based Functional environments to standard Env

```{eval-rst}
.. autoclass:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv

    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv.reset
    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv.step
    .. automethod:: gymnasium.envs.functional_jax_env.FunctionalJaxEnv.render
```
