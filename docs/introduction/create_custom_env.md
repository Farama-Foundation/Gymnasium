---
layout: "contents"
title: Create custom env
---

# Create a Custom Environment

This page provides a short outline of how to create custom environments with Gymnasium, for a more [complete tutorial](../tutorials/gymnasium_basics/environment_creation) with rendering, please read [basic usage](basic_usage)  before reading this page.

We will implement a very simplistic game, called ``GridWorldEnv``, consisting of a 2-dimensional square grid of fixed size. The agent can move vertically or horizontally between grid cells in each timestep and the goal of the agent is to navigate to a target on the grid that has been placed randomly at the beginning of the episode.

Basic information about the game
-  Observations provide the location of the target and agent.
-  There are 4 discrete actions in our environment, corresponding to the movements "right", "up", "left", and "down".
-  The environment ends (terminates) when the agent has navigated to the grid cell where the target is located.
-  The agent is only rewarded when it reaches the target, i.e., the reward is one when the agent reaches the target and zero otherwise.

## Environment `__init__`

```{eval-rst}
.. py:currentmodule:: gymnasium

Like all environments, our custom environment will inherit from :class:`gymnasium.Env` that defines the structure of environment. One of the requirements for an environment is defining the observation and action space, which declare the general set of possible inputs (actions) and outputs (observations) of the environment. As outlined in our basic information about the game, our agent has four discrete actions, therefore we will use the ``Discrete(4)`` space with four options.
```

```{eval-rst}
.. py:currentmodule:: gymnasium.spaces

For our observation, there are a couple options, for this tutorial we will imagine our observation looks like ``{"agent": array([1, 0]), "target": array([0, 3])}`` where the array elements represent the x and y positions of the agent or target. Alternative options for representing the observation is as a 2d grid  with values representing the agent and target on the grid or a 3d grid with each "layer" containing only the agent or target information. Therefore, we will declare the observation space as :class:`Dict` with the agent and target spaces being a :class:`Box` allowing an array output of an int type.
```

For a full list of possible spaces to use with an environment, see [spaces](../api/spaces)

```python
from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid
        self.size = size

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }
```

## Constructing Observations

```{eval-rst}
.. py:currentmodule:: gymnasium

Since we will need to compute observations both in :meth:`Env.reset` and :meth:`Env.step`, it is often convenient to have a method ``_get_obs`` that translates the environment's state into an observation. However, this is not mandatory and you can compute the observations in :meth:`Env.reset` and :meth:`Env.step` separately.
```

```python
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
```

```{eval-rst}
.. py:currentmodule:: gymnasium

We can also implement a similar method for the auxiliary information that is returned by :meth:`Env.reset` and :meth:`Env.step`. In our case, we would like to provide the manhattan distance between the agent and the target:
```

```python
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
```

```{eval-rst}
.. py:currentmodule:: gymnasium

Oftentimes, info will also contain some data that is only available inside the :meth:`Env.step` method (e.g., individual reward terms). In that case, we would have to update the dictionary that is returned by ``_get_info`` in :meth:`Env.step`.
```

## Reset function

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

As the purpose of :meth:`reset` is to initiate a new episode for an environment and has two parameters: ``seed`` and ``options``. The seed can be used to initialize the random number generator to a deterministic state and options can be used to specify values used within reset. On the first line of the reset, you need to call ``super().reset(seed=seed)`` which will initialize the random number generate (:attr:`np_random`) to use through the rest of the :meth:`reset`.

Within our custom environment, the :meth:`reset` needs to randomly choose the agent and target's positions (we repeat this if they have the same position). The return type of :meth:`reset` is a tuple of the initial observation and any auxiliary information. Therefore, we can use the methods ``_get_obs`` and ``_get_info`` that we implemented earlier for that:
```

```python
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
```

## Step function

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

The :meth:`step` method usually contains most of the logic for your environment, it accepts an ``action`` and computes the state of the environment after the applying the action, returning a tuple of the next observation, the resulting reward, if the environment has terminated, if the environment has truncated and auxiliary information.
```
```{eval-rst}
.. py:currentmodule:: gymnasium

For our environment, several things need to happen during the step function:

 - We use the self._action_to_direction to convert the discrete action (e.g., 2) to a grid direction with our agent location. To prevent the agent from going out of bounds of the grd, we clip the agen't location to stay within bounds.
 - We compute the agent's reward by checking if the agent's current position is equal to the target's location.
 - Since the environment doesn't truncate internally (we can apply a time limit wrapper to the environment during :meth:make), we permanently set truncated to False.
 - We once again use _get_obs and _get_info to obtain the agent's observation and auxiliary information.
```

```python
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
```

## Registering and making the environment

```{eval-rst}
While it is possible to use your new custom environment now immediately, it is more common for environments to be initialized using :meth:`gymnasium.make`. In this section, we explain how to register a custom environment then initialize it.

The environment ID consists of three components, two of which are  optional: an optional namespace (here: ``gymnasium_env``), a mandatory  name (here: ``GridWorld``) and an optional but recommended version (here: v0). It may have also be registered as ``GridWorld-v0`` (the  recommended approach), ``GridWorld`` or ``gymnasium_env/GridWorld``, and  the appropriate ID should then be used during environment creation.

The entry point can be a string or function, as this tutorial isn't part of a python project, we cannot use a string but for most environments, this is the normal way of specifying the entry point.

Register has additionally parameters that can be used to specify keyword arguments to the environment, e.g., if to apply a time limit wrapper, etc. See :meth:`gymnasium.register` for more information.
```

```python
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)
```

For a more complete guide on registering a custom environment (including with a string entry point), please read the full [create environment](../tutorials/gymnasium_basics/environment_creation) tutorial.

```{eval-rst}
Once the environment is registered, you can check via :meth:`gymnasium.pprint_registry` which will output all registered environment, and the environment can then be initialized using :meth:`gymnasium.make`. A vectorized version of the environment with multiple instances of the same environment running in parallel can be instantiated with :meth:`gymnasium.make_vec`.
```

```python
import gymnasium as gym
>>> gym.make("gymnasium_env/GridWorld-v0")
<OrderEnforcing<PassiveEnvChecker<GridWorld<gymnasium_env/GridWorld-v0>>>>
>>> gym.make("gymnasium_env/GridWorld-v0", max_episode_steps=100)
<TimeLimit<OrderEnforcing<PassiveEnvChecker<GridWorld<gymnasium_env/GridWorld-v0>>>>>
>>> env = gym.make("gymnasium_env/GridWorld-v0", size=10)
>>> env.unwrapped.size
10
>>> gym.make_vec("gymnasium_env/GridWorld-v0", num_envs=3)
SyncVectorEnv(gymnasium_env/GridWorld-v0, num_envs=3)
```

## Using Wrappers

Oftentimes, we want to use different variants of a custom environment, or we want to modify the behavior of an environment that is provided by Gymnasium or some other party. Wrappers allow us to do this without changing the environment implementation or adding any boilerplate code. Check out the [wrapper documentation](../api/wrappers) for details on how to use wrappers and instructions for implementing your own. In our example, observations cannot be used directly in learning code because they are dictionaries. However, we don't actually need to touch our environment implementation to fix this! We can simply add a wrapper on top of environment instances to flatten observations into a single array:

```python
>>> from gymnasium.wrappers import FlattenObservation

>>> env = gym.make('gymnasium_env/GridWorld-v0')
>>> env.observation_space
Dict('agent': Box(0, 4, (2,), int64), 'target': Box(0, 4, (2,), int64))
>>> env.reset()
({'agent': array([4, 1]), 'target': array([2, 4])}, {'distance': 5.0})
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space
Box(0, 4, (4,), int64)
>>> wrapped_env.reset()
(array([3, 0, 2, 1]), {'distance': 2.0})
```
