
# List of Wrappers

Gymnasium provides a number of commonly used wrappers listed below. More information can be found on the particular
wrapper in the page on the wrapper type

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

.. list-table::
    :header-rows: 1

    * - Name
      - Description
    * - :class:`ArrayConversion`
      - Wraps an environment based on any Array API compatible framework, e.g. ``numpy``, ``torch``, ``jax.numpy``, such that it can be interacted with any other Array API compatible framework.
    * - :class:`AtariPreprocessing`
      - Implements the common preprocessing techniques for Atari environments (excluding frame stacking).
    * - :class:`Autoreset`
      - The wrapped environment is automatically reset when an terminated or truncated state is reached.
    * - :class:`ClipAction`
      - Clips the ``action`` pass to ``step`` to be within the environment's `action_space`.
    * - :class:`ClipReward`
      - Clips the rewards for an environment between an upper and lower bound.
    * - :class:`DelayObservation`
      - Adds a delay to the returned observation from the environment.
    * - :class:`DtypeObservation`
      - Modifies the dtype of an observation array to a specified dtype.
    * - :class:`FilterObservation`
      - Filters a Dict or Tuple observation spaces by a set of keys or indexes.
    * - :class:`FlattenObservation`
      - Flattens the environment's observation space and each observation from ``reset`` and ``step`` functions.
    * - :class:`FrameStackObservation`
      - Stacks the observations from the last ``N`` time steps in a rolling manner.
    * - :class:`GrayscaleObservation`
      - Converts an image observation computed by ``reset`` and ``step`` from RGB to Grayscale.
    * - :class:`HumanRendering`
      - Allows human like rendering for environments that support "rgb_array" rendering.
    * - :class:`JaxToNumpy`
      - Wraps a Jax-based environment such that it can be interacted with NumPy arrays.
    * - :class:`JaxToTorch`
      - Wraps a Jax-based environment so that it can be interacted with PyTorch Tensors.
    * - :class:`MaxAndSkipObservation`
      - Skips the N-th frame (observation) and return the max values between the two last observations.
    * - :class:`NormalizeObservation`
      - Normalizes observations to be centered at the mean with unit variance.
    * - :class:`NormalizeReward`
      - Normalizes immediate rewards such that their exponential moving average has a fixed variance.
    * - :class:`NumpyToTorch`
      - Wraps a NumPy-based environment such that it can be interacted with PyTorch Tensors.
    * - :class:`OrderEnforcing`
      - Will produce an error if ``step`` or ``render`` is called before ``reset``.
    * - :class:`PassiveEnvChecker`
      - A passive environment checker wrapper that surrounds the ``step``, ``reset`` and ``render`` functions to check they follows gymnasium's API.
    * - :class:`RecordEpisodeStatistics`
      - This wrapper will keep track of cumulative rewards and episode lengths.
    * - :class:`RecordVideo`
      - Records videos of environment episodes using the environment's render function.
    * - :class:`RenderCollection`
      - Collect rendered frames of an environment such ``render`` returns a ``list[RenderedFrame]``.
    * - :class:`AddRenderObservation`
      - Includes the rendered observations in the environment's observations.
    * - :class:`RescaleAction`
      - Affinely (linearly) rescales a ``Box`` action space of the environment to within the range of ``[min_action, max_action]``.
    * - :class:`RescaleObservation`
      - Affinely (linearly) rescales a ``Box`` observation space of the environment to within the range of ``[min_obs, max_obs]``.
    * - :class:`ReshapeObservation`
      - Reshapes Array based observations to a specified shape.
    * - :class:`ResizeObservation`
      - Resizes image observations using OpenCV to a specified shape.
    * - :class:`StickyAction`
      - Adds a probability that the action is repeated for the same ``step`` function.
    * - :class:`TimeAwareObservation`
      - Augment the observation with the number of time steps taken within an episode.
    * - :class:`TimeLimit`
      - Limits the number of steps for an environment through truncating the environment if a maximum number of timesteps is exceeded.
    * - :class:`TransformAction`
      - Applies a function to the ``action`` before passing the modified value to the environment ``step`` function.
    * - :class:`TransformObservation`
      - Applies a function to the ``observation`` received from the environment's ``reset`` and ``step`` that is passed back to the user.
    * - :class:`TransformReward`
      - Applies a function to the ``reward`` received from the environment's ``step``.
```

## Vector only Wrappers

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers.vector

.. list-table::
    :header-rows: 1

    * - Name
      - Description
    * - :class:`DictInfoToList`
      - Converts infos of vectorized environments from ``dict`` to ``List[dict]``.
    * - :class:`VectorizeTransformAction`
      - Vectorizes a single-agent transform action wrapper for vector environments.
    * - :class:`VectorizeTransformObservation`
      - Vectorizes a single-agent transform observation wrapper for vector environments.
    * - :class:`VectorizeTransformReward`
      - Vectorizes a single-agent transform reward wrapper for vector environments.
```
