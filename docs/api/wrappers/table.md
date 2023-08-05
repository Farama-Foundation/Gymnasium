
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
      - Implements the common preprocessing techniques for Atari environments (excluding frame stacking).
    * - :class:`AutoresetV0`
      - The wrapped environment is automatically reset when an terminated or truncated state is reached.
    * - :class:`ClipActionV0`
      - Clips the ``action`` pass to ``step`` to be within the environment's `action_space`.
    * - :class:`ClipRewardV0`
      - Clips the rewards for an environment between an upper and lower bound.
    * - :class:`DelayObservationV0`
      - Adds a delay to the returned observation from the environment.
    * - :class:`DtypeObservationV0`
      - Modifies the dtype of an observation array to a specified dtype.
    * - :class:`FilterObservationV0`
      - Filters a Dict or Tuple observation spaces by a set of keys or indexes.
    * - :class:`FlattenObservationV0`
      - Flattens the environment's observation space and each observation from ``reset`` and ``step`` functions.
    * - :class:`FrameStackObservationV0`
      - Stacks the observations from the last ``N`` time steps in a rolling manner.
    * - :class:`GrayscaleObservationV0`
      - Converts an image observation computed by ``reset`` and ``step`` from RGB to Grayscale.
    * - :class:`HumanRenderingV0`
      - Allows human like rendering for environments that support "rgb_array" rendering.
    * - :class:`JaxToNumpyV0`
      - Wraps a Jax-based environment such that it can be interacted with NumPy arrays.
    * - :class:`JaxToTorchV0`
      - Wraps a Jax-based environment so that it can be interacted with PyTorch Tensors.
    * - :class:`LambdaActionV0`
      - Applies a function to the ``action`` before passing the modified value to the environment ``step`` function.
    * - :class:`LambdaObservationV0`
      - Applies a function to the ``observation`` received from the environment's ``reset`` and ``step`` that is passed back to the user.
    * - :class:`LambdaRewardV0`
      - Applies a function to the ``reward`` received from the environment's ``step``.
    * - :class:`MaxAndSkipObservationV0`
      - Skips the N-th frame (observation) and return the max values between the two last observations.
    * - :class:`NormalizeObservationV0`
      - Normalizes observations to be centered at the mean with unit variance.
    * - :class:`NormalizeRewardV1`
      - Normalizes immediate rewards such that their exponential moving average has a fixed variance.
    * - :class:`NumpyToTorchV0`
      - Wraps a NumPy-based environment such that it can be interacted with PyTorch Tensors.
    * - :class:`OrderEnforcingV0`
      - Will produce an error if ``step`` or ``render`` is called before ``render``.
    * - :class:`PassiveEnvCheckerV0`
      - A passive environment checker wrapper that surrounds the ``step``, ``reset`` and ``render`` functions to check they follows gymnasium's API.
    * - :class:`RecordEpisodeStatisticsV0`
      - This wrapper will keep track of cumulative rewards and episode lengths.
    * - :class:`RecordVideoV0`
      - Records videos of environment episodes using the environment's render function.
    * - :class:`RenderCollectionV0`
      - Collect rendered frames of an environment such ``render`` returns a ``list[RenderedFrame]``.
    * - :class:`RenderObservationV0`
      - Includes the rendered observations in the environment's observations.
    * - :class:`RescaleActionV0`
      - Affinely (linearly) rescales a ``Box`` action space of the environment to within the range of ``[min_action, max_action]``.
    * - :class:`RescaleObservationV0`
      - Affinely (linearly) rescales a ``Box`` observation space of the environment to within the range of ``[min_obs, max_obs]``.
    * - :class:`ReshapeObservationV0`
      - Reshapes Array based observations to a specified shape.
    * - :class:`ResizeObservationV0`
      - Resizes image observations using OpenCV to a specified shape.
    * - :class:`StickyActionV0`
      - Adds a probability that the action is repeated for the same ``step`` function.
    * - :class:`TimeAwareObservationV0`
      - Augment the observation with the number of time steps taken within an episode.
    * - :class:`TimeLimitV0`
      - Limits the number of steps for an environment through truncating the environment if a maximum number of timesteps is exceeded.

```

## Vector only Wrappers

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers.vector

.. list-table::
    :header-rows: 1

    * - Name
      - Description
    * - :class:`DictInfoToListV0`
      - Converts infos of vectorized environments from ``dict`` to ``List[dict]``.
    * - :class:`VectorizeLambdaActionV0`
      - Vectorizes a single-agent lambda action wrapper for vector environments.
    * - :class:`VectorizeLambdaObservationV0`
      - Vectori`es a single-agent lambda observation wrapper for vector environments.
    * - :class:`VectorizeLambdaRewardV0`
      - Vectorizes a single-agent lambda reward wrapper for vector environments.

```
