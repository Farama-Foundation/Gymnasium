---
title: Wrapper
---

# Wrappers

```{toctree}
:hidden:

wrappers/table
wrappers/misc_wrappers
wrappers/action_wrappers
wrappers/observation_wrappers
wrappers/reward_wrappers
```

```{eval-rst}
.. automodule:: gymnasium.wrappers
```


```{eval-rst}
.. autoclass:: gymnasium.Wrapper
```

## Methods
```{eval-rst}
.. automethod:: gymnasium.Wrapper.step
.. automethod:: gymnasium.Wrapper.reset
.. automethod:: gymnasium.Wrapper.render
.. automethod:: gymnasium.Wrapper.close
.. automethod:: gymnasium.Wrapper.wrapper_spec
.. automethod:: gymnasium.Wrapper.get_wrapper_attr
.. automethod:: gymnasium.Wrapper.set_wrapper_attr
```

## Attributes
```{eval-rst}
.. autoattribute:: gymnasium.Wrapper.env

    The environment (one level underneath) this wrapper.

    This may itself be a wrapped environment. To obtain the environment underneath all layers of wrappers, use :attr:`gymnasium.Wrapper.unwrapped`.

.. autoproperty:: gymnasium.Wrapper.action_space
.. autoproperty:: gymnasium.Wrapper.observation_space
.. autoproperty:: gymnasium.Wrapper.spec
.. autoproperty:: gymnasium.Wrapper.metadata
.. autoproperty:: gymnasium.Wrapper.np_random
.. autoproperty:: gymnasium.Wrapper.np_random_seed
.. autoproperty:: gymnasium.Wrapper.unwrapped
```
