# List of Gymnasium Wrappers

Gymnasium provides a number of commonly used wrappers listed below. More information can be found on the particular
wrapper in the page on the wrapper type

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

.. list-table::
    :header-rows: 1

    * - Name
      - Description
    * - :class:`AtariPreprocessingV0`
      - Implements the common preprocessing applied tp Atari environments
    * - :class:`AutoresetV0`
      - The wrapped environment will automatically reset when the terminated or truncated  state is reached.
    * - :class:`ClipActionV0`
      - Clip the continuous action to the valid bound specified by the environment's `action_space`
    * - :class:`FilterObservationV0`
      - Filters a dictionary observation spaces to only requested keys
    * - :class:`FlattenObservationV0`
      - An Observation wrapper that flattens the observation
    * - :class:`FrameStackObservationV0`
      - An Observation wrapper that stacks the observations in a rolling manner.
    * - :class:`GrayscaleObservationV0`
      - Convert the image observation from RGB to gray scale.
    * - :class:`HumanRenderingV0`
      - Allows human like rendering for environments that support "rgb_array" rendering
    * - :class:`LambdaActionV0`
      - This wrapper will apply a function to the action before taking a step.
    * - :class:`LambdaObservationV0`
      - This wrapper will apply function to observations
    * - :class:`LambdaRewardV0`
      - This wrapper will apply function to rewards
    * - :class:`NormalizeObservationV0`
      - This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
    * - :class:`NormalizeRewardV1`
      - This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
    * - :class:`OrderEnforcingV0`
      - This will produce an error if `step` or `render` is called before `reset`
    * - :class:`RenderObservationV0`
      - Augment observations by pixel values obtained via `render` that can be added to or replaces the environments observation.
    * - :class:`RecordEpisodeStatisticsV0`
      - This will keep track of cumulative rewards and episode lengths returning them at the end.
    * - :class:`RecordVideoV0`
      - This wrapper will record videos of rollouts.
    * - :class:`RenderCollectionV0`
      - Enable list versions of render modes, i.e. "rgb_array_list" for "rgb_array" such that the rendering for each step are saved in a list until `render` is called.
    * - :class:`RescaleActionV0`
      - Rescales the continuous action space of the environment to a range \[`min_action`, `max_action`], where `min_action` and `max_action` are numpy arrays or floats.
    * - :class:`ResizeObservationV0`
      - This wrapper works on environments with image observations (or more generally observations of shape AxBxC) and resizes the observation to the shape given by the tuple `shape`.
    * - :class:`TimeAwareObservationV0`
      - Augment the observation with current time step in the trajectory (by appending it to the observation).
    * - :class:`TimeLimitV0`
      - This wrapper will emit a truncated signal if the specified number of steps is exceeded in an episode.
```

## Vector only Wrappers

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers.vector

.. list-table::
    :header-rows: 1

    * - Name
      - Description
    * - :class:`DictInfoToList`
      - This wrapper will convert the info of a vectorized environment from the `dict` format to a `list` of dictionaries where the i-th dictionary contains info of the i-th environment.
```
