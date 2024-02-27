---
layout: "contents"
title: Migration Guide
---

# Migration Guide - v0.21 to v1.0.0

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

Gymnasium is a fork of `OpenAI Gym v0.26 <https://github.com/openai/gym/releases/tag/0.26.2>`_, which introduced a large breaking change from `Gym v0.21 <https://github.com/openai/gym/releases/tag/v0.21.0>`_. In this guide, we briefly outline the API changes from Gym v0.21 - which a number of tutorials have been written for - to Gym v0.26. For environments still stuck in the v0.21 API, see the `guide </content/gym_compatibility>`_
```

## Example code for v0.21

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

## Example code for v0.26

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

The ``Env.seed()`` has been removed from the Gym v0.26 environments in favour of ``Env.reset(seed=seed)``. This allows seeding to only be changed on environment reset. The decision to remove ``seed`` was because some environments use emulators that cannot change random number generators within an episode and must be done at the beginning of a new episode. We are aware of cases where controlling the random number generator is important, in these cases, if the environment uses the built-in random number generator, users can set the seed manually with the attribute :attr:`np_random`.

Gymnasium v0.26 changed to using ``numpy.random.Generator`` instead of a custom random number generator. This means that several functions such as ``randint`` were removed in favour of ``integers``. While some environments might use external random number generator, we recommend using the attribute :attr:`np_random` that wrappers and external users can access and utilise.
```

## Environment Reset

```{eval-rst}
In v0.26, :meth:`reset` takes two optional parameters and returns one value. This contrasts to v0.21 which takes no parameters and returns ``None``. The two parameters are ``seed`` for setting the random number generator and ``options`` which allows additional data to be passed to the environment on reset. For example, in classic control, the ``options`` parameter now allows users to modify the range of the state bound. See the original `PR <https://github.com/openai/gym/pull/2921>`_ for more details.

:meth:`reset` further returns ``info``, similar to the ``info`` returned by :meth:`step`. This is important because ``info`` can include metrics or valid action mask that is used or saved in the next step.

To update older environments, we highly recommend that ``super().reset(seed=seed)`` is called on the first line of :meth:`reset`. This will automatically update the :attr:`np_random` with the seed value.
```

## Environment Step

```{eval-rst}
In v0.21, the type definition of :meth:`step` is ``tuple[ObsType, SupportsFloat, bool, dict[str, Any]`` representing the next observation, the reward from the step, if the episode is done and additional info from the step. Due to reproducibility issues that will be expanded on in a blog post soon, we have changed the type definition to ``tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`` adding an extra boolean value. This extra bool corresponds to the older `done` now changed to `terminated` and `truncated`. These changes were introduced in Gym `v0.26 <https://github.com/openai/gym/releases/tag/0.26.0>`_ (turned off by default in `v25 <https://github.com/openai/gym/releases/tag/0.25.0>`_).

For users wishing to update, in most cases, replacing ``done`` with ``terminated`` and ``truncated=False`` in :meth:`step` should address most issues. However, environments that have reasons for episode truncation rather than termination should read through the associated `PR <https://github.com/openai/gym/pull/2752>`_. For users looping through an environment, they should modify ``done = terminated or truncated`` as is show in the example code. For training libraries, the primary difference is to change ``done`` to ``terminated``, indicating whether bootstrapping should or shouldn't happen.
```

## TimeLimit Wrapper
```{eval-rst}
In v0.21, the :class:`TimeLimit` wrapper added an extra key in the ``info`` dictionary ``TimeLimit.truncated`` whenever the agent reached the time limit without reaching a terminal state.

In v0.26, this information is instead communicated through the `truncated` return value described in the previous section, which is `True` if the agent reaches the time limit, whether or not it reaches a terminal state. The old dictionary entry is equivalent to ``truncated and not terminated``
```

## Environment Render

```{eval-rst}
In v0.26, a new render API was introduced such that the render mode is fixed at initialisation as some environments don't allow on-the-fly render mode changes. Therefore, users should now specify the :attr:`render_mode` within ``gym.make`` as shown in the v0.26 example code above.

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
