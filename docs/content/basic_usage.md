---
layout: "contents"
title: API
firstpage:
---

# Basic Usage

## Initializing Environments
Initializing environments is very easy in Gymnasium and can be done via: 

```python
import gymnasium as gym
env = gym.make('CartPole-v1')
```

## Interacting with the Environment
Gymnasium implements the classic "agent-environment loop":

```{image} /_static/diagrams/AE_loop.png
:width: 50%
:align: center
:class: only-light
```

```{image} /_static/diagrams/AE_loop_dark.png
:width: 50%
:align: center
:class: only-dark
```

The agent performs some actions in the environment (usually by passing some control inputs to the environment, e.g. torque inputs of motors) and observes
how the environment's state changes. One such action-observation exchange is referred to as a *timestep*. 

The goal in RL is to manipulate the environment in some specific way. For instance, we want the agent to navigate a robot
to a specific point in space. If it succeeds in doing this (or makes some progress towards that goal), it will receive a positive reward
alongside the observation for this timestep. The reward may also be negative or 0, if the agent did not yet succeed (or did not make any progress). 
The agent will then be trained to maximize the reward it accumulates over many timesteps.

After some timesteps, the environment may enter a terminal state. For instance, the robot may have crashed, or the agent may have succeeded in completing a task. In that case, we want to reset the environment to a new initial state. The environment issues a terminated signal to the agent if it enters such a terminal state. Sometimes we also want to end the episode after a fixed number of timesteps, in this case, the environment issues a truncated signal.
This is a new change in API (v0.26 onwards). Earlier a commonly done signal was issued for an episode ending via any means. This is now changed in favour of issuing two signals - terminated and truncated.

Let's see what the agent-environment loop looks like in Gymnasium.
This example will run an instance of `LunarLander-v2` environment for 1000 timesteps. Since we pass `render_mode="human"`, you should see a window pop up rendering the environment.

```python
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

The output should look something like this:

```{figure} https://user-images.githubusercontent.com/15806078/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif
:width: 50%
:align: center
```

Every environment specifies the format of valid actions by providing an `env.action_space` attribute. Similarly,
the format of valid observations is specified by `env.observation_space`.
In the example above we sampled random actions via `env.action_space.sample()`. Note that we need to seed the action space separately from the 
environment to ensure reproducible samples.


### Change in env.step API

Previously, the step method returned only one boolean - `done`. This is being deprecated in favour of returning two booleans `terminated` and `truncated` (v0.26 onwards). 

`terminated` signal is set to `True` when the core environment terminates inherently because of task completion, failure etc. a condition defined in the MDP.   
`truncated` signal is set to `True` when the episode ends specifically because of a time-limit or a condition not inherent to the environment (not defined in the MDP). 
It is possible for `terminated=True` and `truncated=True` to occur at the same time when termination and truncation occur at the same step. 

This is explained in detail in the `Handling Time Limits` section. 

#### Backward compatibility
Gym will retain support for the old API through compatibility wrappers. 

Users can toggle the old API through `make` by setting `apply_api_compatibility=True`. 

```python
env = gym.make("CartPole-v1", apply_api_compatibility=True)
```
This can also be done explicitly through a wrapper: 
```python
from gymnasium.wrappers import StepAPICompatibility
env = StepAPICompatibility(CustomEnv(), output_truncation_bool=False)
```
For more details see the wrappers section. 


## Checking API-Conformity
If you have implemented a custom environment and would like to perform a sanity check to make sure that it conforms to 
the API, you can run: 

```python
>>> from gymnasium.utils.env_checker import check_env
>>> check_env(env)
```

This function will throw an exception if it seems like your environment does not follow the Gymnasium API. It will also produce
warnings if it looks like you made a mistake or do not follow a best practice (e.g. if `observation_space` looks like 
an image but does not have the right dtype). Warnings can be turned off by passing `warn=False`. By default, `check_env` will 
not check the `render` method. To change this behavior, you can pass `skip_render_check=False`.

> After running `check_env` on an environment, you should not reuse the instance that was checked, as it may have already
been closed!

## Spaces
Spaces are usually used to specify the format of valid actions and observations.
Every environment should have the attributes `action_space` and `observation_space`, both of which should be instances
of classes that inherit from `Space`.
There are multiple `Space` types available in Gymnasium:

- `Box`: describes an n-dimensional continuous space. It's a bounded space where we can define the upper and lower limits which describe the valid values our observations can take.
- `Discrete`: describes a discrete space where {0, 1, ..., n-1} are the possible values our observation or action can take. Values can be shifted to {a, a+1, ..., a+n-1} using an optional argument.
- `Dict`: represents a dictionary of simple spaces.
- `Tuple`: represents a tuple of simple spaces.
- `MultiBinary`: creates a n-shape binary space. Argument n can be a number or a `list` of numbers.
- `MultiDiscrete`: consists of a series of `Discrete` action spaces with a different number of actions in each element.

```python
>>> from gymnasium.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete
>>> import numpy as np 
>>>
>>> observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
>>> observation_space.sample()
[ 1.6952509 -0.4399011 -0.7981693]
>>>
>>> observation_space = Discrete(4)
>>> observation_space.sample()
1
>>> 
>>> observation_space = Discrete(5, start=-2)
>>> observation_space.sample()
-2
>>> 
>>> observation_space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
>>> observation_space.sample()
OrderedDict([('position', 0), ('velocity', 1)])
>>>
>>> observation_space = Tuple((Discrete(2), Discrete(3)))
>>> observation_space.sample()
(1, 2)
>>>
>>> observation_space = MultiBinary(5)
>>> observation_space.sample()
[1 1 1 0 1]
>>>
>>> observation_space = MultiDiscrete([ 5, 2, 2 ])
>>> observation_space.sample()
[3 0 0]
 ```

## Wrappers
Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly.
Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular. Wrappers can 
also be chained to combine their effects. Most environments that are generated via `gymnasium.make` will already be wrapped by default.

In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along
with (possibly optional) parameters to the wrapper's constructor:
```python
>>> import gymnasium
>>> from gymnasium.wrappers import RescaleAction
>>> base_env = gymnasium.make("BipedalWalker-v3")
>>> base_env.action_space
Box([-1. -1. -1. -1.], [1. 1. 1. 1.], (4,), float32)
>>> wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
>>> wrapped_env.action_space
Box([0. 0. 0. 0.], [1. 1. 1. 1.], (4,), float32)
```


There are three very common things you might want a wrapper to do:

- Transform actions before applying them to the base environment
- Transform observations that are returned by the base environment
- Transform rewards that are returned by the base environment

Such wrappers can be easily implemented by inheriting from `ActionWrapper`, `ObservationWrapper`, or `RewardWrapper` and implementing the
respective transformation.

However, sometimes you might need to implement a wrapper that does some more complicated modifications (e.g. modify the
reward based on data in `info`). Such wrappers
can be implemented by inheriting from `Wrapper`.
Gymnasium already provides many commonly used wrappers for you. Some examples:

- `TimeLimit`: Issue a truncated signal if a maximum number of timesteps has been exceeded (or the base environment has issued a truncated signal).
- `ClipAction`: Clip the action such that it lies in the action space (of type `Box`).
- `RescaleAction`: Rescale actions to lie in a specified interval
- `TimeAwareObservation`: Add information about the index of timestep to observation. In some cases helpful to ensure that transitions are Markov.

If you have a wrapped environment, and you want to get the unwrapped environment underneath all of the layers of wrappers (so that you can manually call a function or change some underlying aspect of the environment), you can use the `.unwrapped` attribute. If the environment is already a base environment, the `.unwrapped` attribute will just return itself.

```python
>>> wrapped_env
<RescaleAction<TimeLimit<BipedalWalker<BipedalWalker-v3>>>>
>>> wrapped_env.unwrapped
<gymnasium.envs.box2d.bipedal_walker.BipedalWalker object at 0x7f87d70712d0>
```

## Playing within an environment
You can also play the environment using your keyboard using the `play` function in `gymnasium.utils.play`. 
```python
from gymnasium.utils.play import play
play(gymnasium.make('Pong-v0'))
```
This opens a window of the environment and allows you to control the agent using your keyboard.

Playing using the keyboard requires a key-action map. This map should have type `dict[tuple[int], int | None]`, which maps the keys pressed to action performed.
For example, if pressing the keys `w` and `space` at the same time is supposed to perform action `2`, then the `key_to_action` dict should look like this:
```python
{
    # ...
    (ord('w'), ord(' ')): 2,
    # ...
}
```
As a more complete example, let's say we wish to play with `CartPole-v0` using our left and right arrow keys. The code would be as follows:
```python
import gymnasium as gym
import pygame
from gymnasium.utils.play import play

mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
play(gym.make("CartPole-v1",render_mode="rgb_array"), keys_to_action=mapping)
```
where we obtain the corresponding key ID constants from pygame. If the `key_to_action` argument is not specified, then the default `key_to_action` mapping for that env is used, if provided.

Furthermore, if you wish to plot real time statistics as you play, you can use `gymnasium.utils.play.PlayPlot`. Here's some sample code for plotting the reward for last 5 second of gameplay:
```python
import gymnasium as gym
import pygame
from gymnasium.utils.play import PlayPlot, play

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    return [rew, ]

plotter = PlayPlot(callback, 30 * 5, ["reward"])
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
env = gym.make("CartPole-v1", render_mode="rgb_array")
play(env, callback=plotter.callback, keys_to_action=mapping)
```
