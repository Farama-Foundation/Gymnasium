---
title: Wrapper
---

# Wrappers

```{toctree}
:hidden:
wrappers/misc_wrappers
wrappers/action_wrappers
wrappers/observation_wrappers
wrappers/reward_wrappers
```

```{eval-rst}
.. automodule:: gymnasium.wrappers

```

## gymnasium.Wrapper

```{eval-rst}
.. autoclass:: gymnasium.Wrapper
```

### Methods

```{eval-rst}
.. autofunction:: gymnasium.Wrapper.step
.. autofunction:: gymnasium.Wrapper.reset
.. autofunction:: gymnasium.Wrapper.close
```

### Attributes

```{eval-rst}
.. autoproperty:: gymnasium.Wrapper.action_space
.. autoproperty:: gymnasium.Wrapper.observation_space
.. autoproperty:: gymnasium.Wrapper.reward_range
.. autoproperty:: gymnasium.Wrapper.spec
.. autoproperty:: gymnasium.Wrapper.metadata
.. autoproperty:: gymnasium.Wrapper.np_random
.. attribute:: gymnasium.Wrapper.env

    The environment (one level underneath) this wrapper.

    This may itself be a wrapped environment.
    To obtain the environment underneath all layers of wrappers, use :attr:`gymnasium.Wrapper.unwrapped`.

.. autoproperty:: gymnasium.Wrapper.unwrapped
```

## Gymnasium Wrappers

Gymnasium provides a number of commonly used wrappers listed below. More information can be found on the particular
wrapper in the page on the wrapper type

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

.. list-table::
    :header-rows: 1

    * - Name
      - Type
      - Description
    * - :class:`AtariPreprocessing`
      - Misc Wrapper
      - Implements the common preprocessing applied tp Atari environments
    * - :class:`AutoResetWrapper`
      - Misc Wrapper
      - The wrapped environment will automatically reset when the terminated or truncated  state is reached.
    * - :class:`ClipAction`
      - Action Wrapper
      - Clip the continuous action to the valid bound specified by the environment's `action_space`
    * - :class:`EnvCompatibility`
      - Misc Wrapper
      - Provides compatibility for environments written in the OpenAI Gym v0.21 API to look like Gymnasium environments
    * - :class:`FilterObservation`
      - Observation Wrapper
      - Filters a dictionary observation spaces to only requested keys
    * - :class:`FlattenObservation`
      - Observation Wrapper
      - An Observation wrapper that flattens the observation
    * - :class:`FrameStack`
      - Observation Wrapper
      - AnObservation wrapper that stacks the observations in a rolling manner.
    * - :class:`GrayScaleObservation`
      - Observation Wrapper
      - Convert the image observation from RGB to gray scale.
    * - :class:`HumanRendering`
      - Misc Wrapper
      - Allows human like rendering for environments that support "rgb_array" rendering
    * - :class:`NormalizeObservation`
      - Observation Wrapper
      - This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
    * - :class:`NormalizeReward`
      - Reward Wrapper
      - This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
    * - :class:`OrderEnforcing`
      - Misc Wrapper
      - This will produce an error if `step` or `render` is called before `reset`
    * - :class:`PixelObservationWrapper`
      - Observation Wrapper
      - Augment observations by pixel values obtained via `render` that can be added to or replaces the environments observation.
    * - :class:`RecordEpisodeStatistics`
      - Misc Wrapper
      - This will keep track of cumulative rewards and episode lengths returning them at the end.
    * - :class:`RecordVideo`
      - Misc Wrapper
      - This wrapper will record videos of rollouts.
    * - :class:`RenderCollection`
      - Misc Wrapper
      - Enable list versions of render modes, i.e. "rgb_array_list" for "rgb_array" such that the rendering for each step are saved in a list until `render` is called.
    * - :class:`RescaleAction`
      - Action Wrapper
      - Rescales the continuous action space of the environment to a range \[`min_action`, `max_action`], where `min_action` and `max_action` are numpy arrays or floats.
    * - :class:`ResizeObservation`
      - Observation Wrapper
      - This wrapper works on environments with image observations (or more generally observations of shape AxBxC) and resizes the observation to the shape given by the tuple `shape`.
    * - :class:`StepAPICompatibility`
      - Misc Wrapper
      - Modifies an environment step function from (old) done to the (new) termination / truncation API.
    * - :class:`TimeAwareObservation`
      - Observation Wrapper
      - Augment the observation with current time step in the trajectory (by appending it to the observation).
    * - :class:`TimeLimit`
      - Misc Wrapper
      - This wrapper will emit a truncated signal if the specified number of steps is exceeded in an episode.
    * - :class:`TransformObservation`
      - Observation Wrapper
      - This wrapper will apply function to observations
    * - :class:`TransformReward`
      - Reward Wrapper
      - This wrapper will apply function to rewards
    * - :class:`VectorListInfo`
      - Misc Wrapper
      - This wrapper will convert the info of a vectorized environment from the `dict` format to a `list` of dictionaries where the i-th dictionary contains info of the i-th environment.
```
