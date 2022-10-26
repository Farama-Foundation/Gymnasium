---
layout: "contents"
title: Compatibility With Gym
---

# Compatibility with Gym

Gymnasium provides a number of compatibility methods for a range of Environment implementations. 

## Loading OpenAI Gym environments

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

For environments that are registered solely in OpenAI Gym, it is still possible to import environments within Gymnasium however they will not appear in the gymnasium environment registry. Introduced in Gymnasium v0.26.3, using the special environment ``"GymV26Environment-v0"``, passing an ``env_name`` along with any other keyword will be passed to ``gym.make``. This environment, :class:`EnvCompatibility`, is also compatibility with passing gym environment instances with the ``env`` keyword. 
```

An example of this is atari 0.8.0 which does not have a gymnasium implementation. 
```python
import gymnasium as gym
env = gym.make("GymV26Environment-v0", env_id="ALE/Pong-v5")
```

## Gym v0.22 Environment Compatibility

```{eval-rst}
.. py:currentmodule:: gymnasium

A number of environments have not updated to the recent Gym changes, in particular since v0.21. Therefore, to increase backward compatibility, Gym and Gymnasium v0.26+ include an ``apply_api_compatibility`` in :meth:`make` parameter that applies a wrappers to convert v0.21 environment to the v0.26 API.
```

```python
import gym
env = gym.make("OldV21Env-v0", apply_api_compatibility=True)
```

## Step API Compatibility 

```{eval-rst}
If environments implement the (old) done step API, Gymnasium provides both functions (:meth:`gymnasium.utils.step_api_compatibility.convert_to_terminated_truncated_step_api`) and wrappers (:class:`gymnasium.wrappers.StepAPICompatibility`) that will convert the step function to the (new) termination and truncation step API. 
```
