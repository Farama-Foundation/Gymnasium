---
layout: "contents"
title: Migration Guide
---

# v21 to v26 Migration Guide

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

Gymnasium is a fork of `OpenAI Gym v26 <https://github.com/openai/gym/releases/tag/0.26.2>`_, therefore, this requires environments and training libraries to update to the v26 API. In this guide, we briefly outline the changes to the `Gym v21 <https://github.com/openai/gym/releases/tag/v0.21.0>`_ API which a number of tutorials and environment have been written in. For environments that have not updated, users can use the :class:`EnvCompatibility` wrapper, for more information see the `guide </content/gym_compatibility>`_
```

### Example code for v21
```python
import gym
env = gym.make("LunarLander-v2", options={})
env.seed(123)
observation = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, done, info = env.step(action)
    
    env.render(mode="human")

env.close()
```

### Example code for v26
```python
import gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=123, options={})

done = False
while not done:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

env.close()
```

## Seed and random number generator

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

The ``Env.seed()`` has been removed from the Gym v26 environments in favour of ``Env.reset(seed=seed)`` as a majority of the time, users call seed just before resetting the environment. This decision to remove ``seed`` was due to environment that use emulators often cannot change the random number generator within an episode and must be done at the beginning of a new episode. We are aware of cases where controlling the random number generator is important, in these cases, if the environment uses the built in random number generator, users can set :attr:`np_random`. 

Gymnasium v26 changed to using the ``numpy.random.Generator`` instead of a custom random number generator. This means that several functions such as ``randint`` were removed in favour of ``integers``. While some environments might use external random number generator, we recommend using the :attr:`np_random` that wrappers and external users can access and utilise. 
```

## Environment Reset

```{eval-rst}
In v26, :meth:`reset` has two new parameters compared to v21 along with an extra return value. The two parmeters of ``reset`` are ``seed`` for setting the random number generator with the second parameter being ``options`` allowing additional data to be passed to the environment on reset. For example, in the classic control, the options parameter now allows users to modify the range of the state bound. See the original `PR <https://github.com/openai/gym/pull/2921>`_ for more details. 

For the new return value, ``info``, this is similar to the ``info`` returned by the :meth:`step`. This is important for ``info`` can include metrics or valid action mask that is used or saved in the next step. 

To update environments, we highly recommend that the first line of the environment :meth:`reset` function is ``super().reset(seed=seed)`` which will automatically update the :attr:`np_random` with the seed value. 
```

## Environment Step 

```{eval-rst}
In v21, the type definition of :meth:`step` is ``tuple[ObsType, SupportsFloat, bool, dict[str, Any]`` representing the next observation, the reward from the step, if the episode is done and additional info from the step. Due to reproductibility issues that will be expanded on a blog post soon to be published, we have changed the type definition to ``tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`` adding an extra boolean value. This is as the ``done`` value has been replaced with two variables, `terminated` and `truncated`. These changes were introduced in Gym `v26 <https://github.com/openai/gym/releases/tag/0.26.0>`_ (turned off by default in `v25 <https://github.com/openai/gym/releases/tag/0.25.0>`_). 

For users wishing to update, in most cases, replacing ``done`` with ``terminated`` and ``truncated=False`` in environment should address most environments. However, for environments that have custom reasons for an episode to truncate rather than terminate should read through the associated `PR <https://github.com/openai/gym/pull/2752>`_. For users looping through an environment, they should modify ``done = terminated or truncated`` as is show in the example code. For training libraries, primarily requires changing ``done`` to ``terminated``, indicating that bootstraping should or shouldn't happen.  
```

## Environment Render

```{eval-rst}
In v26, a new render API was introduced such that the render mode is fixed at initialisation as some environment don't allow the render mode to change on fly or allow the render mode to pre-compute data at initialisation. Therefore, users should now specify the render mode within ``gym.make`` as shown in the v26 example code above. 

For a more complete explanation of the changes, please refer to this `summary <https://younis.dev/blog/render-api/>`_.
```

## Removed code

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

* GoalEnv - This was removed, users needing it should reimplement the environment or use Gymnasium Robotics which contains an implementation of this environment.
* ``from gym.envs.classic_control import rendering`` - This was removed in favour of users implementing their own rendering systems. Gymnasium environments are coded using pygame. 
* Robotics environments - The robotics environments have been moved to the `Gymnasium Robotics <https://robotics.farama.org/>`_ project. 
* Monitor wrapper - This wrapper was replaced with two separate wrapper, :class:`RecordVideo` and :class:`RecordEpisodeStatistics`
```
