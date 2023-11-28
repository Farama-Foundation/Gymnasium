---
layout: "contents"
title: Basic Usage
firstpage:
---

# Basic Usage

```{eval-rst}
.. py:currentmodule:: gymnasium

Gymnasium is a project that provides an API (application programming interface) for all single agent reinforcement learning environments with implementations of common environments: cartpole, pendulum, mountain-car, mujoco, atari, and more. This page will outline the basics of how to use Gymnasium including its four key functions: :meth:`make`, :meth:`Env.reset`, :meth:`Env.step` and :meth:`Env.render`.

At the core of Gymnasium is :class:`Env`, a high-level python class representing a markov decision process (MDP) from reinforcement learning theory (note: this is not a perfect reconstruction, missing several components of MDPs). The class provides users the ability generate an initial state, transition / move to new states given an action and the visualise the environment. Alongside :class:`Env`, :class:`Wrapper` are provided to help augment / modify the environment, in particular, the agent observations, rewards and actions taken.
```

## Initializing Environments

```{eval-rst}
.. py:currentmodule:: gymnasium

Initializing environments is very easy in Gymnasium and can be done via the :meth:`make` function:
```

```python
import gymnasium as gym
env = gym.make('CartPole-v1')
```

```{eval-rst}
.. py:currentmodule:: gymnasium

This function will return an :class:`Env` for users to interact with. To see all environments you can create, use :meth:`pprint_registry`. Furthermore, :meth:`make` provides a number of additional arguments for specifying keywords to the environment, adding more or less wrappers, etc. See :meth:`make` for more information.
```

## Interacting with the Environment

Within reinforcement learning, the classic "agent-environment loop" pictured below is simplified representation of how an agent and environment interact with each other. The agent receives an observation about the environment, the agent then selects an action that the environment uses to determine the reward and the next observation. The cycle then repeating itself until the environment ends (terminates).

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

For gymnasium, the "agent-environment-loop" is implemented below for a single episode (until the environment ends). See the next section for a line-by-line explanation. Note that running this code requires install swig (`pip install swig` or [download](https://www.swig.org/download.html)) along with `pip install gymnasium[box2d]`.

```python
import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
```

The output should look something like this:

```{figure} https://user-images.githubusercontent.com/15806078/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif
:width: 50%
:align: center
```

### Explaining the code

```{eval-rst}
.. py:currentmodule:: gymnasium

First, an environment is created using :meth:`make` with an additional keyword ``"render_mode"`` that specifies how the environment should be visualised. See :meth:`Env.render` for details on the default meaning of different render modes. In this example, we use the ``"LunarLander"`` environment where the agent controls a spaceship that needs to land safely.

After initializing the environment, we :meth:`Env.reset` the environment to get the first observation of the environment along with an additional information. For initializing the environment with a particular random seed or options (see the environment documentation for possible values) use the ``seed`` or ``options`` parameters with :meth:`reset`.

As we wish to continue the agent-environment loop until the environment ends, which is in an unknown number of timesteps, we define ``episode_over`` as a variable to know when to stop interacting with the environment along with a while loop that uses it.

Next, the agent performs an action in the environment, :meth:`Env.step` executes the select actions (in this case random with ``env.action_space.sample()``) to update the environment. This action can be imagined as moving a robot or pressing a button on a games' controller that causes a change within the environment. As a result, the agent receives a new observation from the updated environment along with a reward for taking the action. This reward could be for instance positive for destroying an enemy or a negative reward for moving into lava. One such action-observation exchange is referred to as a **timestep**.

However, after some timesteps, the environment may end, this is called the terminal state. For instance, the robot may have crashed, or may have succeeded in completing a task, the environment will need to stop as the agent cannot continue. In gymnasium, if the environment has terminated, this is returned by :meth:`step` as the third variable, ``terminated``. Similarly, we may also want the environment to end after a fixed number of timesteps, in this case, the environment issues a truncated signal. If either of ``terminated`` or ``truncated`` are ``True`` then we end the episode but in most cases users might wish to restart the environment, this can be done with `env.reset()`.
```

## Action and observation spaces

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

Every environment specifies the format of valid actions and observations with the :attr:`action_space` and :attr:`observation_space` attributes. This is helpful for both knowing the expected input and output of the environment as all valid actions and observation should be contained with their respective space. In the example above, we sampled random actions via ``env.action_space.sample()`` instead of using an agent policy, mapping observations to actions which users will want to make.

Importantly, :attr:`Env.action_space` and :attr:`Env.observation_space` are instances of :class:`Space`, a high-level python class that provides the key functions: :meth:`Space.contains` and :meth:`Space.sample`. Gymnasium has support for a wide range of spaces that users might need:

.. py:currentmodule:: gymnasium.spaces

- :class:`Box`: describes bounded space with upper and lower limits of any n-dimensional shape.
- :class:`Discrete`: describes a discrete space where ``{0, 1, ..., n-1}`` are the possible values our observation or action can take.
- :class:`MultiBinary`: describes a binary space of any n-dimensional shape.
- :class:`MultiDiscrete`: consists of a series of :class:`Discrete` action spaces with a different number of actions in each element.
- :class:`Text`: describes a string space with a minimum and maximum length
- :class:`Dict`: describes a dictionary of simpler spaces.
- :class:`Tuple`: describes a tuple of simple spaces.
- :class:`Graph`: describes a mathematical graph (network) with interlinking nodes and edges
- :class:`Sequence`: describes a variable length of simpler space elements.

For example usage of spaces, see their `documentation <../api/spaces>`_ along with `utility functions <../api/spaces/utils>`_. There are a couple of more niche spaces :class:`Graph`, :class:`Sequence` and :class:`Text`.
```

## Modifying the environment

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly. Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular. Wrappers can also be chained to combine their effects. Most environments that are generated via :meth:`gymnasium.make` will already be wrapped by default using the :class:`TimeLimit`, :class:`OrderEnforcing` and :class:`PassiveEnvChecker`.

In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along with (possibly optional) parameters to the wrapper's constructor:
```

```python
>>> import gymnasium as gym
>>> from gymnasium.wrappers import FlattenObservation
>>> env = gym.make("CarRacing-v2")
>>> env.observation_space.shape
(96, 96, 3)
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space.shape
(27648,)
```

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

Gymnasium already provides many commonly used wrappers for you. Some examples:

- :class:`TimeLimit`: Issues a truncated signal if a maximum number of timesteps has been exceeded (or the base environment has issued a truncated signal).
- :class:`ClipAction`: Clips any action passed to ``step`` such that it lies in the base environment's action space.
- :class:`RescaleAction`: Applies an affine transformation to the action to linearly scale for a new low and high bound on the environment.
- :class:`TimeAwareObservation`: Add information about the index of timestep to observation. In some cases helpful to ensure that transitions are Markov.
```

For a full list of implemented wrappers in gymnasium, see [wrappers](/api/wrappers).

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

If you have a wrapped environment, and you want to get the unwrapped environment underneath all the layers of wrappers (so that you can manually call a function or change some underlying aspect of the environment), you can use the :attr:`unwrapped` attribute. If the environment is already a base environment, the :attr:`unwrapped` attribute will just return itself.
```

```python
>>> wrapped_env
<FlattenObservation<TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v2>>>>>>
>>> wrapped_env.unwrapped
<gymnasium.envs.box2d.car_racing.CarRacing object at 0x7f04efcb8850>
```

## More information

* [Training an agent](train_agent)
* [Making a Custom Environment](create_custom_env)
* [Recording an agent's behaviour](record_agent)
* [Speeding up an Environment](speed_up_env)
* [Compatibility with OpenAI Gym](gym_compatibility)
* [Migration Guide for Gym v0.21 to v0.26 and for v1.0.0](migration_guide)
