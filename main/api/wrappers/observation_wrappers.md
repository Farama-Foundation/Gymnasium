# Observation Wrappers

### *class* gymnasium.ObservationWrapper(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType])

Modify observations from [`Env.reset()`](../env.md#gymnasium.Env.reset) and [`Env.step()`](../env.md#gymnasium.Env.step) using [`observation()`](#gymnasium.ObservationWrapper.observation) function.

If you would like to apply a function to only the observation before
passing it to the learning code, you can simply inherit from [`ObservationWrapper`](#gymnasium.ObservationWrapper) and overwrite the method
[`observation()`](#gymnasium.ObservationWrapper.observation) to implement that transformation. The transformation defined in that method must be
reflected by the `env` observation space. Otherwise, you need to specify the new observation space of the
wrapper by setting `self.observation_space` in the `__init__()` method of your wrapper.

* **Parameters:**
  **env** – Environment to be wrapped.

#### observation(observation: ObsType) → WrapperObsType

Returns a modified observation.

* **Parameters:**
  **observation** – The `env` observation
* **Returns:**
  **The modified observation**

## Implemented wrappers

### *class* gymnasium.wrappers.TransformObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], func: Callable[[ObsType], Any], observation_space: [Space](../spaces.md#gymnasium.spaces.Space)[WrapperObsType] | None)

Applies a function to the `observation` received from the environment’s `Env.reset()` and `Env.step()` that is passed back to the user.

The function `func` will be applied to all observations.
If the observations from `func` are outside the bounds of the `env`’s observation space, provide an updated `observation_space`.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.TransformObservation`](../vector/wrappers.md#gymnasium.wrappers.vector.TransformObservation).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import TransformObservation
>>> import numpy as np
>>> np.random.seed(0)
>>> env = gym.make("CartPole-v1")
>>> env.reset(seed=42)
(array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), {})
>>> env = gym.make("CartPole-v1")
>>> env = TransformObservation(env, lambda obs: obs + 0.1 * np.random.random(obs.shape), env.observation_space)
>>> env.reset(seed=42)
(array([0.08227695, 0.06540678, 0.09613613, 0.07422512]), {})
```

Change logs:
: * v0.15.4 - Initially added
  * v1.0.0 - Add requirement of `observation_space`

* **Parameters:**
  * **env** – The environment to wrap
  * **func** – A function that will transform an observation. If this transformed observation is outside the observation space of `env.observation_space` then provide an observation_space.
  * **observation_space** – The observation spaces of the wrapper, if None, then it is assumed the same as `env.observation_space`.

### *class* gymnasium.wrappers.DelayObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], delay: int)

Adds a delay to the returned observation from the environment.

Before reaching the `delay` number of timesteps, returned observations is an array of zeros with
the same shape as the observation space.

No vector version of the wrapper exists.

#### NOTE
This does not support random delay values, if users are interested, please raise an issue or pull request to add this feature.

### Example

```pycon
>>> import gymnasium as gym
>>> env = gym.make("CartPole-v1")
>>> env.reset(seed=123)
(array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})
```

```pycon
>>> env = DelayObservation(env, delay=2)
>>> env.reset(seed=123)
(array([0., 0., 0., 0.], dtype=float32), {})
>>> env.step(env.action_space.sample())
(array([0., 0., 0., 0.], dtype=float32), 1.0, False, False, {})
>>> env.step(env.action_space.sample())
(array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), 1.0, False, False, {})
```

Change logs:
: * v1.0.0 - Initially added

* **Parameters:**
  * **env** – The environment to wrap
  * **delay** – The number of timesteps to delay observations

### *class* gymnasium.wrappers.DtypeObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], dtype: Any)

Modifies the dtype of an observation array to a specified dtype.

#### NOTE
This is only compatible with `Box`, `Discrete`, `MultiDiscrete` and `MultiBinary` observation spaces

A vector version of the wrapper exists [`gymnasium.wrappers.vector.DtypeObservation`](../vector/wrappers.md#gymnasium.wrappers.vector.DtypeObservation).

Change logs:
: * v1.0.0 - Initially added

* **Parameters:**
  * **env** – The environment to wrap
  * **dtype** – The new dtype of the observation, as a NumPy scalar type,
    dtype object, or string

### *class* gymnasium.wrappers.FilterObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], filter_keys: Sequence[str | int])

Filters a Dict or Tuple observation spaces by a set of keys or indexes.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.FilterObservation`](../vector/wrappers.md#gymnasium.wrappers.vector.FilterObservation).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import FilterObservation
>>> env = gym.make("CartPole-v1")
>>> env = gym.wrappers.TimeAwareObservation(env, flatten=False)
>>> env.observation_space
Dict('obs': Box([-4.8               -inf -0.41887903        -inf], [4.8               inf 0.41887903        inf], (4,), float32), 'time': Box(0, 500, (1,), int32))
>>> env.reset(seed=42)
({'obs': array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), 'time': array([0], dtype=int32)}, {})
>>> env = FilterObservation(env, filter_keys=['time'])
>>> env.reset(seed=42)
({'time': array([0], dtype=int32)}, {})
>>> env.step(0)
({'time': array([1], dtype=int32)}, 1.0, False, False, {})
```

Change logs:
: * v0.12.3 - Initially added, originally called FilterObservationWrapper
  * v1.0.0 - Rename to FilterObservation and add support for tuple observation spaces with integer `filter_keys`

* **Parameters:**
  * **env** – The environment to wrap
  * **filter_keys** – The set of subspaces to be *included*, use a list of strings for `Dict` and integers for `Tuple` spaces

### *class* gymnasium.wrappers.FlattenObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType])

Flattens the environment’s observation space and each observation from `reset` and `step` functions.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.FlattenObservation`](../vector/wrappers.md#gymnasium.wrappers.vector.FlattenObservation).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import FlattenObservation
>>> env = gym.make("CarRacing-v3")
>>> env.observation_space.shape
(96, 96, 3)
>>> env = FlattenObservation(env)
>>> env.observation_space.shape
(27648,)
>>> obs, _ = env.reset()
>>> obs.shape
(27648,)
```

Change logs:
: * v0.15.0 - Initially added

* **Parameters:**
  **env** – The environment to wrap

### *class* gymnasium.wrappers.FrameStackObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], stack_size: int, \*, padding_type: str | ObsType = 'reset')

Stacks the observations from the last `N` time steps in a rolling manner.

For example, if the number of stacks is 4, then the returned observation contains
the most recent 4 observations. For environment ‘Pendulum-v1’, the original observation
is an array with shape [3], so if we stack 4 observations, the processed observation
has shape [4, 3].

Users have options for the padded observation used:

> * “reset” (default) - The reset value is repeated
> * “zero” - A “zero”-like instance of the observation space
> * custom - An instance of the observation space

No vector version of the wrapper exists.

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import FrameStackObservation
>>> env = gym.make("CarRacing-v3")
>>> env = FrameStackObservation(env, stack_size=4)
>>> env.observation_space
Box(0, 255, (4, 96, 96, 3), uint8)
>>> obs, _ = env.reset()
>>> obs.shape
(4, 96, 96, 3)
```

Example with different padding observations:
: ```pycon
  >>> env = gym.make("CartPole-v1")
  >>> env.reset(seed=123)
  (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})
  >>> stacked_env = FrameStackObservation(env, 3)   # the default is padding_type="reset"
  >>> stacked_env.reset(seed=123)
  (array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
         [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
         [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
        dtype=float32), {})
  ```
  <br/>
  ```pycon
  >>> stacked_env = FrameStackObservation(env, 3, padding_type="zero")
  >>> stacked_env.reset(seed=123)
  (array([[ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
        dtype=float32), {})
  >>> stacked_env = FrameStackObservation(env, 3, padding_type=np.array([1, -1, 0, 2], dtype=np.float32))
  >>> stacked_env.reset(seed=123)
  (array([[ 1.        , -1.        ,  0.        ,  2.        ],
         [ 1.        , -1.        ,  0.        ,  2.        ],
         [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
        dtype=float32), {})
  ```

Change logs:
: * v0.15.0 - Initially add as `FrameStack` with support for lz4
  * v1.0.0 - Rename to `FrameStackObservation` and remove lz4 and `LazyFrame` support
    : along with adding the `padding_type` parameter

* **Parameters:**
  * **env** – The environment to apply the wrapper
  * **stack_size** – The number of frames to stack.
  * **padding_type** – The padding type to use when stacking the observations, options: “reset”, “zero”, custom obs

### *class* gymnasium.wrappers.GrayscaleObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], keep_dim: bool = False)

Converts an image observation computed by `reset` and `step` from RGB to Grayscale.

The `keep_dim` will keep the channel dimension.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.GrayscaleObservation`](../vector/wrappers.md#gymnasium.wrappers.vector.GrayscaleObservation).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import GrayscaleObservation
>>> env = gym.make("CarRacing-v3")
>>> env.observation_space.shape
(96, 96, 3)
>>> grayscale_env = GrayscaleObservation(env)
>>> grayscale_env.observation_space.shape
(96, 96)
>>> grayscale_env = GrayscaleObservation(env, keep_dim=True)
>>> grayscale_env.observation_space.shape
(96, 96, 1)
```

Change logs:
: * v0.15.0 - Initially added, originally called `GrayScaleObservation`
  * v1.0.0 - Renamed to `GrayscaleObservation`

* **Parameters:**
  * **env** – The environment to wrap
  * **keep_dim** – If to keep the channel in the observation, if `True`, `obs.shape == 3` else `obs.shape == 2`

### *class* gymnasium.wrappers.MaxAndSkipObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], skip: int = 4)

Skips the N-th frame (observation) and return the max values between the two last observations.

If the episode terminates or is truncated before `skip` steps, the maximum
is taken over the last two observations received during that call. If only
one step is taken, that step’s observation is returned unchanged.

No vector version of the wrapper exists.

#### NOTE
This wrapper is based on the wrapper from [stable-baselines3]([https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#MaxAndSkipEnv](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#MaxAndSkipEnv))

### Example

```pycon
>>> import gymnasium as gym
>>> env = gym.make("CartPole-v1")
>>> obs0, *_ = env.reset(seed=123)
>>> obs1, *_ = env.step(1)
>>> obs2, *_ = env.step(1)
>>> obs3, *_ = env.step(1)
>>> obs4, *_ = env.step(1)
>>> skip_and_max_obs = np.max(np.stack([obs3, obs4], axis=0), axis=0)
>>> env = gym.make("CartPole-v1")
>>> wrapped_env = MaxAndSkipObservation(env)
>>> wrapped_obs0, *_ = wrapped_env.reset(seed=123)
>>> wrapped_obs1, *_ = wrapped_env.step(1)
>>> np.all(obs0 == wrapped_obs0)
np.True_
>>> np.all(wrapped_obs1 == skip_and_max_obs)
np.True_
```

Change logs:
: * v1.0.0 - Initially add

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The environment to apply the wrapper
  * **skip** – The number of frames to skip

### *class* gymnasium.wrappers.NormalizeObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], epsilon: float = 1e-8)

Normalizes observations to be centered at the mean with unit variance.

The property `update_running_mean` allows to freeze/continue the running mean calculation of the observation
statistics. If `True` (default), the `RunningMeanStd` will get updated every time `step` or `reset` is called.
If `False`, the calculated statistics are used but not updated anymore; this may be used during evaluation.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.NormalizeObservation`](../vector/wrappers.md#gymnasium.wrappers.vector.NormalizeObservation).

#### NOTE
The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
newly instantiated or the policy was changed recently.

### Example

```pycon
>>> import numpy as np
>>> import gymnasium as gym
>>> env = gym.make("CartPole-v1")
>>> obs, info = env.reset(seed=123)
>>> term, trunc = False, False
>>> while not (term or trunc):
...     obs, _, term, trunc, _ = env.step(1)
...
>>> obs
array([ 0.1511158 ,  1.7183299 , -0.25533703, -2.8914354 ], dtype=float32)
>>> env = gym.make("CartPole-v1")
>>> env = NormalizeObservation(env)
>>> obs, info = env.reset(seed=123)
>>> term, trunc = False, False
>>> while not (term or trunc):
...     obs, _, term, trunc, _ = env.step(1)
>>> obs
array([ 2.0059888,  1.5676788, -1.9944268, -1.6120394], dtype=float32)
```

Change logs:
: * v0.21.0 - Initially add
  * v1.0.0 - Add update_running_mean attribute to allow disabling of updating the running mean / standard, particularly useful for evaluation time.
    : Casts all observations to np.float32 and sets the observation space with low/high of -np.inf and np.inf and dtype as np.float32

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The environment to apply the wrapper
  * **epsilon** – A stability parameter that is used when scaling the observations.
* **Raises:**
  [**InvalidBound**](../utils.md#gymnasium.error.InvalidBound) – If `epsilon` is not finite and strictly positive.

### *class* gymnasium.wrappers.AddRenderObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], render_only: bool = True, render_key: str = 'pixels', obs_key: str = 'state')

Includes the rendered observations in the environment’s observations.

### Notes

This was previously called `PixelObservationWrapper`.

No vector version of the wrapper exists.

Example - Replace the observation with the rendered image:
: ```pycon
  >>> env = gym.make("CartPole-v1", render_mode="rgb_array")
  >>> env = AddRenderObservation(env, render_only=True)
  >>> env.observation_space
  Box(0, 255, (400, 600, 3), uint8)
  >>> obs, _ = env.reset(seed=123)
  >>> image = env.render()
  >>> np.all(obs == image)
  np.True_
  >>> obs, *_ = env.step(env.action_space.sample())
  >>> image = env.render()
  >>> np.all(obs == image)
  np.True_
  ```

Example - Add the rendered image to the original observation as a dictionary item:
: ```pycon
  >>> env = gym.make("CartPole-v1", render_mode="rgb_array")
  >>> env = AddRenderObservation(env, render_only=False)
  >>> env.observation_space
  Dict('pixels': Box(0, 255, (400, 600, 3), uint8), 'state': Box([-4.8               -inf -0.41887903        -inf], [4.8               inf 0.41887903        inf], (4,), float32))
  >>> obs, info = env.reset(seed=123)
  >>> obs.keys()
  dict_keys(['state', 'pixels'])
  >>> obs["state"]
  array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32)
  >>> np.all(obs["pixels"] == env.render())
  np.True_
  >>> obs, reward, terminates, truncates, info = env.step(env.action_space.sample())
  >>> image = env.render()
  >>> np.all(obs["pixels"] == image)
  np.True_
  ```

Change logs:
: * v0.15.0 - Initially added as `PixelObservationWrapper`
  * v1.0.0 - Renamed to `AddRenderObservation`

* **Parameters:**
  * **env** – The environment to wrap.
  * **render_only** (*bool*) – If `True` (default), the original observation returned
    by the wrapped environment will be discarded, and a dictionary
    observation will only include pixels. If `False`, the
    observation dictionary will contain both the original
    observations and the pixel observations.
  * **render_key** – Optional custom string specifying the pixel key. Defaults to “pixels”
  * **obs_key** – Optional custom string specifying the obs key. Defaults to “state”.
    Must differ from `render_key` when `render_only=False` and the
    original observation space is not a `spaces.Dict`.

### *class* gymnasium.wrappers.ResizeObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], shape: tuple[int, int])

Resizes image observations using OpenCV to a specified shape.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.ResizeObservation`](../vector/wrappers.md#gymnasium.wrappers.vector.ResizeObservation).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import ResizeObservation
>>> env = gym.make("CarRacing-v3")
>>> env.observation_space.shape
(96, 96, 3)
>>> resized_env = ResizeObservation(env, (32, 32))
>>> resized_env.observation_space.shape
(32, 32, 3)
```

Change logs:
: * v0.12.6 - Initially added
  * v1.0.0 - Requires `shape` with a tuple of two integers

* **Parameters:**
  * **env** – The environment to wrap
  * **shape** – The resized observation shape

### *class* gymnasium.wrappers.ReshapeObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], shape: int | tuple[int, ...])

Reshapes Array based observations to a specified shape.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.RescaleObservation`](../vector/wrappers.md#gymnasium.wrappers.vector.RescaleObservation).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import ReshapeObservation
>>> env = gym.make("CarRacing-v3")
>>> env.observation_space.shape
(96, 96, 3)
>>> reshape_env = ReshapeObservation(env, (24, 4, 96, 1, 3))
>>> reshape_env.observation_space.shape
(24, 4, 96, 1, 3)
```

Change logs:
: * v1.0.0 - Initially added

* **Parameters:**
  * **env** – The environment to wrap
  * **shape** – The reshaped observation space

### *class* gymnasium.wrappers.RescaleObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], min_obs: floating | integer | ndarray, max_obs: floating | integer | ndarray)

Affinely (linearly) rescales a `Box` observation space of the environment to within the range of `[min_obs, max_obs]`.

For unbounded components in the original observation space, the corresponding target bounds must also be infinite and vice versa.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.RescaleObservation`](../vector/wrappers.md#gymnasium.wrappers.vector.RescaleObservation).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import RescaleObservation
>>> env = gym.make("Pendulum-v1")
>>> env.observation_space
Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
>>> env = RescaleObservation(env, np.array([-2, -1, -10], dtype=np.float32), np.array([1, 0, 1], dtype=np.float32))
>>> env.observation_space
Box([ -2.  -1. -10.], [1. 0. 1.], (3,), float32)
```

Change logs:
: * v1.0.0 - Initially added

* **Parameters:**
  * **env** – The environment to wrap
  * **min_obs** – The new minimum observation bound
  * **max_obs** – The new maximum observation bound

### *class* gymnasium.wrappers.TimeAwareObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], flatten: bool = True, normalize_time: bool = False, \*, dict_time_key: str = 'time')

Augment the observation with the number of time steps taken within an episode.

The `normalize_time` if `True` represents time as a normalized value between [0,1]
otherwise if `False`, the current timestep is an integer.

For environments with `Dict` observation spaces, the time information is automatically
added in the key “time” (can be changed through `dict_time_key`) and for environments with `Tuple`
observation space, the time information is added as the final element in the tuple.
Otherwise, the observation space is transformed into a `Dict` observation space with two keys,
“obs” for the base environment’s observation and “time” for the time information.

To flatten the observation, use the `flatten` parameter which will use the
[`gymnasium.spaces.utils.flatten()`](../spaces/utils.md#gymnasium.spaces.utils.flatten) function.

No vector version of the wrapper exists.

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import TimeAwareObservation
>>> env = gym.make("CartPole-v1")
>>> env = TimeAwareObservation(env)
>>> env.observation_space
Box([-4.80000019        -inf -0.41887903        -inf  0.        ], [4.80000019e+00            inf 4.18879032e-01            inf
 5.00000000e+02], (5,), float64)
>>> env.reset(seed=42)[0]
array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ,  0.        ])
>>> _ = env.action_space.seed(42)
>>> env.step(env.action_space.sample())[0]
array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476,  1.        ])
```

Normalize time observation space example:
: ```pycon
  >>> env = gym.make('CartPole-v1')
  >>> env = TimeAwareObservation(env, normalize_time=True)
  >>> env.observation_space
  Box([-4.8               -inf -0.41887903        -inf  0.        ], [4.8               inf 0.41887903        inf 1.        ], (5,), float32)
  >>> env.reset(seed=42)[0]
  array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ,  0.        ],
        dtype=float32)
  >>> _ = env.action_space.seed(42)
  >>> env.step(env.action_space.sample())[0]
  array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476,  0.002     ],
        dtype=float32)
  ```

Flatten observation space example:
: ```pycon
  >>> env = gym.make("CartPole-v1")
  >>> env = TimeAwareObservation(env, flatten=False)
  >>> env.observation_space
  Dict('obs': Box([-4.8               -inf -0.41887903        -inf], [4.8               inf 0.41887903        inf], (4,), float32), 'time': Box(0, 500, (1,), int32))
  >>> env.reset(seed=42)[0]
  {'obs': array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), 'time': array([0], dtype=int32)}
  >>> _ = env.action_space.seed(42)
  >>> env.step(env.action_space.sample())[0]
  {'obs': array([ 0.02727336, -0.20172954,  0.03625453,  0.32351476], dtype=float32), 'time': array([1], dtype=int32)}
  ```

Change logs:
: * v0.18.0 - Initially added
  * v1.0.0 - Remove vector environment support, add `flatten` and `normalize_time` parameters

* **Parameters:**
  * **env** – The environment to apply the wrapper
  * **flatten** – Flatten the observation to a Box of a single dimension
  * **normalize_time** – if True return time in the range [0,1]
    otherwise return time as remaining timesteps before truncation
  * **dict_time_key** – For environment with a `Dict` observation space, the key for the time space. By default, “time”.

### *class* gymnasium.wrappers.DiscretizeObservation(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], bins: int | tuple[int, ...], multidiscrete: bool = False)

Uniformly discretizes a continuous Box observation space into a single Discrete space.

Example 1 - Discretize MountainCar observation space:
: ```pycon
  >>> env = gym.make("MountainCar-v0")
  >>> env.observation_space
  Box([-1.2  -0.07], [0.6  0.07], (2,), float32)
  >>> obs, _ = env.reset(seed=42)
  >>> obs
  array([-0.4452088,  0.       ], dtype=float32)
  >>> env = DiscretizeObservation(env, bins=10)
  >>> env.observation_space
  Discrete(100)
  >>> obs, _ = env.reset(seed=42)
  >>> obs
  45
  ```

Example 2 - Discretize LunarLander observation space:
: ```pycon
  >>> env = gym.make("LunarLander-v3")
  >>> env.observation_space
  Box([ -2.5        -2.5       -10.        -10.         -6.2831855 -10.
    -0.         -0.       ], [ 2.5        2.5       10.        10.         6.2831855 10.
    1.         1.       ], (8,), float32)
  >>> obs, _ = env.reset(seed=42)
  >>> obs.round(3)
  array([ 0.002,  1.418,  0.233,  0.32 , -0.003, -0.053,  0.   ,  0.   ],
        dtype=float32)
  >>> env = DiscretizeObservation(env, bins=3)
  >>> env.observation_space
  Discrete(6561)
  >>> obs, _ = env.reset(seed=42)
  >>> obs
  4005
  ```

Example 3 - Discretize LunarLander observation space with MultiDiscrete:
: ```pycon
  >>> env = gym.make("LunarLander-v3")
  >>> env.observation_space
  Box([ -2.5        -2.5       -10.        -10.         -6.2831855 -10.
    -0.         -0.       ], [ 2.5        2.5       10.        10.         6.2831855 10.
    1.         1.       ], (8,), float32)
  >>> obs, _ = env.reset(seed=42)
  >>> obs.round(3)
  array([ 0.002,  1.418,  0.233,  0.32 , -0.003, -0.053,  0.   ,  0.   ],
        dtype=float32)
  >>> env = DiscretizeObservation(env, bins=3, multidiscrete=True)
  >>> env.observation_space
  MultiDiscrete([3 3 3 3 3 3 3 3])
  >>> obs, _ = env.reset(seed=42)
  >>> obs
  array([1, 2, 1, 1, 1, 1, 0, 0])
  ```

* **Parameters:**
  * **env** – The environment to wrap.
  * **bins** – int or tuple of ints (number of bins per dimension).
  * **multidiscrete** – If True, use MultiDiscrete space instead of flattening to Discrete.
