---
title: AsyncVectorEnv
---

# AsyncVectorEnv

```{eval-rst}
.. autoclass:: gymnasium.vector.AsyncVectorEnv

    .. automethod:: gymnasium.vector.AsyncVectorEnv.reset
    .. automethod:: gymnasium.vector.AsyncVectorEnv.step
    .. automethod:: gymnasium.vector.AsyncVectorEnv.close

    .. automethod:: gymnasium.vector.AsyncVectorEnv.call
    .. automethod:: gymnasium.vector.AsyncVectorEnv.get_attr
    .. automethod:: gymnasium.vector.AsyncVectorEnv.set_attr
```

## Asynchronous methods

{meth}`~gymnasium.vector.AsyncVectorEnv.reset`, {meth}`~gymnasium.vector.AsyncVectorEnv.step` and {meth}`~gymnasium.vector.AsyncVectorEnv.call` are each implemented as a pair of methods: an `_async` method that sends the command to the worker processes and returns immediately, and a `_wait` method that blocks until the results are returned. These can be used directly to run other code while the sub-environments step.

```{eval-rst}
.. automethod:: gymnasium.vector.AsyncVectorEnv.reset_async
.. automethod:: gymnasium.vector.AsyncVectorEnv.reset_wait
.. automethod:: gymnasium.vector.AsyncVectorEnv.step_async
.. automethod:: gymnasium.vector.AsyncVectorEnv.step_wait
.. automethod:: gymnasium.vector.AsyncVectorEnv.call_async
.. automethod:: gymnasium.vector.AsyncVectorEnv.call_wait
```

## Additional methods

```{eval-rst}
.. autoproperty:: gymnasium.vector.AsyncVectorEnv.np_random
.. autoproperty:: gymnasium.vector.AsyncVectorEnv.np_random_seed
```
