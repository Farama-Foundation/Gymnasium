---
layout: "contents"
title: Compatibility With Gym
---

# Compatibility with Gym

Gymnasium provides a number of compatibility methods for a range of Environment implementations.

## Loading OpenAI Gym environments

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

For environments that are registered solely in OpenAI Gym and not in Gymnasium, Gymnasium v0.26.3 and above allows importing them through either a special environment or a wrapper. The ``"GymV26Environment-v0"`` environment was introduced in Gymnasium v0.26.3, and allows importing of Gym environments through the ``env_name`` argument along with other relevant kwargs environment kwargs. To perform conversion through a wrapper, the environment itself can be passed to the wrapper :class:`EnvCompatibility` through the ``env`` kwarg.
```

An example of this is atari 0.8.0 which does not have a gymnasium implementation.
```python
import gymnasium as gym

env = gym.make("GymV26Environment-v0", env_id="ALE/Pong-v5")
```

## Gym v0.21 Environment Compatibility

```{eval-rst}
.. py:currentmodule:: gymnasium

A number of environments have not updated to the recent Gym changes, in particular since v0.21. This update is significant for the introduction of ``termination`` and ``truncation`` signatures in favour of the previously used ``done``. To allow backward compatibility, Gym and Gymnasium v0.26+ include an ``apply_api_compatibility`` kwarg when calling :meth:`make` that automatically converts a v0.21 API compliant environment to one that is compatible with v0.26+.
```

```python
import gym

env = gym.make("OldV21Env-v0", apply_api_compatibility=True)
```

Additionally, in Gymnasium we provide specialist environments for compatibility that for ``env_id`` will call ``gym.make``.
```python
import gymnasium

env = gymnasium.make("GymV21Environment-v0", env_id="CartPole-v1", render_mode="human")
# or
env = gymnasium.make("GymV21Environment-v0", env=OldV21Env())

```

## Step API Compatibility

```{eval-rst}
If environments implement the (old) done step API, Gymnasium provides both functions (:meth:`gymnasium.utils.step_api_compatibility.convert_to_terminated_truncated_step_api`) and wrappers (:class:`gymnasium.wrappers.StepAPICompatibility`) that will convert an environment with the old step API (using ``done``) to the new step API (using ``termination`` and ``truncation``).
```
