---
title: Registry
---

# Registry

Gymnasium allows users to automatically load environments, pre-wrapped with several important wrappers.
Environments can also be created through python imports.

## Make

```{eval-rst}
.. autofunction:: gymnasium.make
```

## Register

```{eval-rst}
.. autofunction:: gymnasium.register
```

## All registered environments

To find all the registered Gymnasium environments, use the `gymnasium.pprint_registry()`.
This will not include environments registered only in OpenAI Gym however can be loaded by `gymnasium.make`.

## Spec

```{eval-rst}
.. autofunction:: gymnasium.spec
```

## Pretty print registry

```{eval-rst}
.. autofunction:: gymnasium.pprint_registry
```
