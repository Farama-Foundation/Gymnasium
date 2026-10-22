---
title: Registry
---

# Make and register

```{eval-rst}
Gymnasium allows users to automatically load environments, pre-wrapped with several important wrappers through the :func:`gymnasium.make` function. To do this, the environment must be registered prior with :func:`gymnasium.register`. To get the environment specifications for a registered environment, use :func:`gymnasium.spec` and to print the whole registry, use :func:`gymnasium.pprint_registry`.
For environments registered by a third-party package on import, :func:`gymnasium.register_envs` can be used so that IDEs and linters don't treat the import as unused.

.. autofunction:: gymnasium.make
.. autofunction:: gymnasium.make_vec
.. autofunction:: gymnasium.register
.. autofunction:: gymnasium.register_envs
.. autofunction:: gymnasium.spec
.. autofunction:: gymnasium.pprint_registry
```

## Core variables

```{eval-rst}
.. autoclass:: gymnasium.envs.registration.EnvSpec

    .. automethod:: gymnasium.envs.registration.EnvSpec.make
    .. automethod:: gymnasium.envs.registration.EnvSpec.to_json
    .. automethod:: gymnasium.envs.registration.EnvSpec.from_json
    .. automethod:: gymnasium.envs.registration.EnvSpec.pprint

.. autoclass:: gymnasium.envs.registration.WrapperSpec
.. autoclass:: gymnasium.VectorizeMode
    :members:
    :undoc-members:
.. attribute:: gymnasium.envs.registration.registry

    The Global registry for gymnasium which is where environment specifications are stored by :func:`gymnasium.register` and from which :func:`gymnasium.make` is used to create environments.

.. attribute:: gymnasium.envs.registration.current_namespace

    The current namespace when creating or registering environments. This is by default ``None`` but with :func:`~gymnasium.envs.registration.namespace` this can be modified to automatically set the environment id namespace.
```

## Additional functions

```{eval-rst}
.. autofunction:: gymnasium.envs.registration.get_env_id
.. autofunction:: gymnasium.envs.registration.parse_env_id
.. autofunction:: gymnasium.envs.registration.find_highest_version
.. autofunction:: gymnasium.envs.registration.namespace
.. autofunction:: gymnasium.envs.registration.load_env_creator
```
