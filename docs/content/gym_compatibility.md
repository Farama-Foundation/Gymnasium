---
layout: "contents"
title: Compatibility With Gym
---

# Compatibility with Gym
Gymnasium provides a number of compatibility methods for a range of Environment implementations. 

## Loading OpenAI Gym environments
For environments that are registered solely in OpenAI Gym, it is still possible to import environments within Gymnasium.
Introduced in Gymnasium v0.26.3, in `gymnasium.make`, if the environment id doesn't exist in the 
gymnasium registry then a check if done to see if the environment is registered in Gym. 
If it is, then we will make environment using the gym's registration information. 

An example of this is atari 0.8.0 which does not have a gymnasium implementation currently. 
```python
import gymnasium as gym
env = gym.make("ALE/Pong-v5")
```

## < v0.22 Environment Compatibility
A number of environments have not updated to the recent Gym changes, in particular since v0.21. 
Therefore, to increase backward compatibility, Gym and Gymnasium v0.26+ include an `apply_api_compatibility`
in `{eval-rst}:py:meth:gymnasium.make` parameter that applies a wrappers to convert v0.21 environment to the v0.26 API.

## Step API Compatibility 
If environments implement the (old) done step API, Gymnasium provides both functions 
(`{eval-rst}:py:meth:gymnasium.utils.step_api_compatibility.convert_to_terminated_truncated_step_api`) and 
wrappers (`{eval-rst}:py:meth:gymnasium.wrappers.StepAPICompatibility`) that will convert the 
step function to the (new) termination and truncation step API. 
