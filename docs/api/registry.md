---
title: Registry
---

# Make and register

```{eval-rst}
Gymnasium allows users to automatically load environments, pre-wrapped with several important wrappers through the :meth:`gymnasium.make` function. To do this, the environment must be registered prior with :meth:`gymnasium.register`. To get the environment specifications for a registered environment, use :meth:`gymnasium.spec` and to print the whole registry, use :meth:`gymnasium.pprint_registry`.

.. autofunction:: gymnasium.make
.. autofunction:: gymnasium.make_vec
.. autofunction:: gymnasium.register
.. autofunction:: gymnasium.spec
.. autofunction:: gymnasium.pprint_registry
```

## Core variables

```{eval-rst}
.. autoclass:: gymnasium.envs.registration.EnvSpec
.. autoclass:: gymnasium.envs.registration.WrapperSpec
.. attribute:: gymnasium.envs.registration.registry

    The Global registry for gymnasium which is where environment specifications are stored by :meth:`gymnasium.register` and from which :meth:`gymnasium.make` is used to create environments.

.. attribute:: gymnasium.envs.registration.current_namespace

    The current namespace when creating or registering environments. This is by default ``None`` by with :meth:`namespace` this can be modified to automatically set the environment id namespace.
```

## Additional functions

```{eval-rst}
.. autofunction:: gymnasium.envs.registration.get_env_id
.. autofunction:: gymnasium.envs.registration.parse_env_id
.. autofunction:: gymnasium.envs.registration.find_highest_version
.. autofunction:: gymnasium.envs.registration.namespace
.. autofunction:: gymnasium.envs.registration.load_env_creator
```
