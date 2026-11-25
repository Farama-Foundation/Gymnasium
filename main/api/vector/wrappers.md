# Wrappers

<a id="vector-wrappers-autoreset-support"></a>

## Autoreset mode support

Vector wrappers don’t all support every [autoreset mode](autoreset.md), read from the wrapped environment’s `metadata["autoreset_mode"]`. Wrappers raise a `ValueError` at construction for an unsupported mode.

| Wrapper                                                                                                                                                                                                     | Next-step   | Same-step                                                            | Disabled                                                                          |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|----------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `VectorizeTransformObservation` and its subclasses (`FilterObservation`, `FlattenObservation`, `GrayscaleObservation`, `ResizeObservation`, `ReshapeObservation`, `RescaleObservation`, `DtypeObservation`) | ✓           | ✓ (`info["final_obs"]` is transformed for each sub-environment)      | ✓                                                                                 |
| `TransformObservation` and other `VectorObservationWrapper` subclasses                                                                                                                                      | ✓           | ✗ (`info["final_obs"]` is not transformed)                           | ✓                                                                                 |
| `NormalizeObservation`                                                                                                                                                                                      | ✓           | ✓ (`info["final_obs"]` is normalized and included in the statistics) | ✓ (only resets of all sub-environments, partial `reset_mask` resets are rejected) |
| Action wrappers (`VectorActionWrapper` subclasses, including `VectorizeTransformAction`)                                                                                                                    | ✓           | ✓                                                                    | ✓                                                                                 |
| Reward wrappers (`VectorRewardWrapper` subclasses, including `VectorizeTransformReward`)                                                                                                                    | ✓           | ✓                                                                    | ✓                                                                                 |
| `NormalizeReward`                                                                                                                                                                                           | ✓           | ✓                                                                    | ✓                                                                                 |
| `RecordEpisodeStatistics`                                                                                                                                                                                   | ✓           | ✓                                                                    | ✓                                                                                 |
| `DictInfoToList`                                                                                                                                                                                            | ✓           | ✓                                                                    | ✓                                                                                 |
| `RecordVideo`                                                                                                                                                                                               | ✓           | ✓ (recorded episodes don’t include the final frame)                  | ✓                                                                                 |

Unlike the single-environment `NormalizeReward`, the vector `NormalizeReward` restarts a sub-environment’s accumulated return once it is reset, i.e., at the end of an episode for same-step autoreset and on `reset` for disabled autoreset.

For next-step autoreset, reward wrappers return the reward of a sub-environment’s autoreset step unmodified, as no environment step occurs.

### *class* gymnasium.vector.VectorWrapper(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType])

Wraps the vectorized environment to allow a modular transformation.

This class is the base class for all wrappers for vectorized environments. The subclass
could override some methods to change the behavior of the original vectorized environment
without touching the original code.

Equivalent to [`gymnasium.Wrapper`](../wrappers.md#gymnasium.Wrapper) for vectorized environments, with the same type parameters,
`VectorWrapper[WrapperObsType, WrapperActType, ObsType, ActType, ArrayType]`, where `WrapperObsType` and
`WrapperActType` are the types the wrapper exposes and `ObsType` and `ActType` are the wrapped environment’s.
The reward and termination/truncation `ArrayType` is shared between the wrapper and the wrapped environment.

#### NOTE
Don’t forget to call `super().__init__(env)` if the subclass overrides `__init__()`.

* **Parameters:**
  **env** – The environment to wrap

#### step(actions: WrapperActType) → tuple[WrapperObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]

Step through all environments using the actions returning the batched data.

#### reset(\*, seed: int | None = None, options: dict[str, Any] | None = None) → tuple[WrapperObsType, dict[str, Any]]

Reset all environment using seed and options.

#### render() → tuple[str | ndarray[Any, Any] | tuple[ndarray[Any, Any], ndarray[Any, Any]], ...] | None

Returns the render mode from the base vector environment.

#### close(\*\*kwargs: Any) → None

Close all environments.

### *class* gymnasium.vector.VectorObservationWrapper(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType])

Wraps the vectorized environment to allow a modular transformation of the observation.

Equivalent to [`gymnasium.ObservationWrapper`](../wrappers/observation_wrappers.md#gymnasium.ObservationWrapper) for vectorized environments.

Same-step autoreset is not supported as `info["final_obs"]` is not transformed.

* **Parameters:**
  **env** – Vector environment.

#### observations(observations: ObsType) → WrapperObsType

Defines the vector observation transformation.

* **Parameters:**
  **observations** – A vector observation from the environment
* **Returns:**
  **the transformed observation**

### *class* gymnasium.vector.VectorActionWrapper(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType])

Wraps the vectorized environment to allow a modular transformation of the actions.

Equivalent of [`gymnasium.ActionWrapper`](../wrappers/action_wrappers.md#gymnasium.ActionWrapper) for vectorized environments.

* **Parameters:**
  **env** – Vector environment.

#### actions(actions: WrapperActType) → ActType

Transform the actions before sending them to the environment.

* **Parameters:**
  **actions** (*WrapperActType*) – the actions to transform
* **Returns:**
  **ActType** – the transformed actions

### *class* gymnasium.vector.VectorRewardWrapper(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType])

Wraps the vectorized environment to allow a modular transformation of the reward.

Equivalent of [`gymnasium.RewardWrapper`](../wrappers/reward_wrappers.md#gymnasium.RewardWrapper) for vectorized environments.

* **Parameters:**
  **env** – Vector environment.

#### rewards(rewards: ArrayType) → ArrayType

Transform the reward before returning it.

* **Parameters:**
  **rewards** (*array*) – the reward to transform
* **Returns:**
  **array** – the transformed reward

## Vector only wrappers

### *class* gymnasium.wrappers.vector.DictInfoToList(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType])

Converts infos of vectorized environments from `dict` to `List[dict]`.

This wrapper converts the info format of a
vector environment from a dictionary to a list of dictionaries.
This wrapper is intended to be used around vectorized
environments. If using other wrappers that perform
operation on info like RecordEpisodeStatistics this
need to be the outermost wrapper.

i.e. `DictInfoToList(RecordEpisodeStatistics(vector_env))`

### Example

```pycon
>>> import numpy as np
>>> dict_info = {
...      "k": np.array([0., 0., 0.5, 0.3]),
...      "_k": np.array([False, False, True, True])
...  }
...
>>> list_info = [{}, {}, {"k": 0.5}, {"k": 0.3}]
```

Example for vector environments:
: ```pycon
  >>> import numpy as np
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("CartPole-v1", num_envs=3)
  >>> obs, info = envs.reset(seed=123)
  >>> info
  {}
  >>> envs = DictInfoToList(envs)
  >>> obs, info = envs.reset(seed=123)
  >>> info
  [{}, {}, {}]
  ```

Another example for vector environments:
: ```pycon
  >>> import numpy as np
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("HalfCheetah-v5", num_envs=2)
  >>> _ = envs.reset(seed=123)
  >>> _ = envs.action_space.seed(123)
  >>> _, _, _, _, infos = envs.step(envs.action_space.sample())
  >>> infos
  {'x_position': array([0.03332211, 0.10172355]), '_x_position': array([ True,  True]), 'x_velocity': array([-0.06296527,  0.89345848]), '_x_velocity': array([ True,  True]), 'reward_forward': array([-0.06296527,  0.89345848]), '_reward_forward': array([ True,  True]), 'reward_ctrl': array([-0.24503504, -0.21944423], dtype=float32), '_reward_ctrl': array([ True,  True])}
  >>> envs = DictInfoToList(envs)
  >>> _ = envs.reset(seed=123)
  >>> _ = envs.action_space.seed(123)
  >>> _, _, _, _, infos = envs.step(envs.action_space.sample())
  >>> infos  
  [{'x_position': np.float64(0.0333221...), 'x_velocity': np.float64(-0.0629652...), 'reward_forward': np.float64(-0.0629652...), 'reward_ctrl': np.float32(-0.2450350...)}, {'x_position': np.float64(0.1017235...), 'x_velocity': np.float64(0.8934584...), 'reward_forward': np.float64(0.8934584...), 'reward_ctrl': np.float32(-0.2194442...)}]
  ```

Change logs:
: * v0.24.0 - Initially added as `VectorListInfo`
  * v1.0.0 - Renamed to `DictInfoToList`

* **Parameters:**
  **env** ([*Env*](../env.md#gymnasium.Env)) – The environment to apply the wrapper

### *class* gymnasium.wrappers.vector.VectorizeTransformObservation(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], wrapper: type[[TransformObservation](../wrappers/observation_wrappers.md#gymnasium.wrappers.TransformObservation)], \*\*kwargs: Any)

Vectorizes a single-agent transform observation wrapper for vector environments.

Most of the lambda observation wrappers for single agent environments have vectorized implementations,
it is advised that users simply use those instead via importing from gymnasium.wrappers.vector….
The following example illustrate use-cases where a custom lambda observation wrapper is required.

Example - The normal observation:
: ```pycon
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
  >>> obs, info = envs.reset(seed=123)
  >>> envs.close()
  >>> obs
  array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
         [ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598],
         [ 0.03517495, -0.000635  , -0.01098382, -0.03203924]],
        dtype=float32)
  ```

Example - Applying a custom lambda observation wrapper that duplicates the observation from the environment
: ```pycon
  >>> import numpy as np
  >>> import gymnasium as gym
  >>> from gymnasium.spaces import Box
  >>> from gymnasium.wrappers import TransformObservation
  >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
  >>> old_space = envs.single_observation_space
  >>> new_space = Box(low=np.array([old_space.low, old_space.low]), high=np.array([old_space.high, old_space.high]))
  >>> envs = VectorizeTransformObservation(envs, wrapper=TransformObservation, func=lambda x: np.array([x, x]), observation_space=new_space)
  >>> obs, info = envs.reset(seed=123)
  >>> envs.close()
  >>> obs
  array([[[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
          [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
  <br/>
         [[ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598],
          [ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598]],
  <br/>
         [[ 0.03517495, -0.000635  , -0.01098382, -0.03203924],
          [ 0.03517495, -0.000635  , -0.01098382, -0.03203924]]],
        dtype=float32)
  ```

* **Parameters:**
  * **env** – The vector environment to wrap.
  * **wrapper** – The wrapper to vectorize
  * **\*\*kwargs** – Keyword argument for the wrapper

### *class* gymnasium.wrappers.vector.VectorizeTransformAction(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], wrapper: type[[TransformAction](../wrappers/action_wrappers.md#gymnasium.wrappers.TransformAction)], \*\*kwargs: Any)

Vectorizes a single-agent transform action wrapper for vector environments.

Example - Without action transformation:
: ```pycon
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
  >>> _ = envs.action_space.seed(123)
  >>> obs, info = envs.reset(seed=123)
  >>> obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
  >>> envs.close()
  >>> obs
  array([[-4.6343064e-01,  9.8971417e-05],
         [-4.4488689e-01, -1.9375233e-03],
         [-4.3118435e-01, -1.5342437e-03]], dtype=float32)
  ```

Example - Adding a transform that applies a ReLU to the action:
: ```pycon
  >>> import gymnasium as gym
  >>> from gymnasium.wrappers import TransformAction
  >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
  >>> envs = VectorizeTransformAction(envs, wrapper=TransformAction, func=lambda x: (x > 0.0) * x, action_space=envs.single_action_space)
  >>> _ = envs.action_space.seed(123)
  >>> obs, info = envs.reset(seed=123)
  >>> obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
  >>> envs.close()
  >>> obs
  array([[-4.6343064e-01,  9.8971417e-05],
         [-4.4354835e-01, -5.9898634e-04],
         [-4.3034542e-01, -6.9532328e-04]], dtype=float32)
  ```

* **Parameters:**
  * **env** – The vector environment to wrap
  * **wrapper** – The wrapper to vectorize
  * **\*\*kwargs** – Arguments for the LambdaAction wrapper

### *class* gymnasium.wrappers.vector.VectorizeTransformReward(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, \_ArrayT], wrapper: type[[TransformReward](../wrappers/reward_wrappers.md#gymnasium.wrappers.TransformReward)], \*\*kwargs: Any)

Vectorizes a single-agent transform reward wrapper for vector environments.

An example such that applies a ReLU to the reward:
: ```pycon
  >>> import gymnasium as gym
  >>> from gymnasium.wrappers import TransformReward
  >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
  >>> envs = VectorizeTransformReward(envs, wrapper=TransformReward, func=lambda x: (x > 0.0) * x)
  >>> _ = envs.action_space.seed(123)
  >>> obs, info = envs.reset(seed=123)
  >>> obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
  >>> envs.close()
  >>> rew
  array([-0., -0., -0.])
  ```

* **Parameters:**
  * **env** – The vector environment to wrap.
  * **wrapper** – The wrapper to vectorize
  * **\*\*kwargs** – Keyword argument for the wrapper

## Vectorized common wrappers

### *class* gymnasium.wrappers.vector.RecordEpisodeStatistics(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ndarray], buffer_length: int = 100, stats_key: str = 'episode')

This wrapper will keep track of cumulative rewards and episode lengths.

At the end of any episode within the vectorized env, the statistics of the episode
will be added to `info` using the key `episode`, and the `_episode` key
is used to indicate the environment index which has a terminated or truncated episode.

```pycon
>>> infos = {  
...     ...
...     "episode": {
...         "r": "<array of cumulative reward for each done sub-environment>",
...         "l": "<array of episode length for each done sub-environment>",
...         "t": "<array of elapsed time since beginning of episode for each done sub-environment>"
...     },
...     "_episode": "<boolean array of length num-envs>"
... }
```

Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
`wrapped_env.return_queue` and `wrapped_env.length_queue` respectively.

* **Variables:**
  * **return_queue** (*deque* *[**np.float64* *]*) – The cumulative rewards of the last `deque_size`-many episodes
  * **length_queue** (deque[

    ```
    np.int_
    ```

    ]) – The lengths of the last `deque_size`-many episodes

### Example

```pycon
>>> from pprint import pprint
>>> import gymnasium as gym
>>> from gymnasium.wrappers.vector import RecordEpisodeStatistics
>>> envs = gym.make_vec("CartPole-v1", num_envs=3)
>>> envs = RecordEpisodeStatistics(envs)
>>> obs, info = envs.reset(123)
>>> _ = envs.action_space.seed(123)
>>> end = False
>>> while not end:
...     obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
...     end = term.any() or trunc.any()
...
>>> envs.close()
>>> pprint(info) 
{'_episode': array([ True, False, False]),
 'episode': {'l': array([11,  0,  0], dtype=int32),
             'r': array([11.,  0.,  0.], dtype=float32),
             't': array([0.007812, 0.      , 0.      ], dtype=float32)},
```

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The environment to apply the wrapper
  * **buffer_length** – The size of the buffers `return_queue`, `length_queue` and `time_queue`
  * **stats_key** – The info key to save the data

### *class* gymnasium.wrappers.vector.RecordVideo(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ndarray], video_folder: str, video_aspect_ratio: tuple[int, int] = (1, 1), record_first_only: bool = False, episode_trigger: Callable[[int], bool] | None = None, step_trigger: Callable[[int], bool] | None = None, video_length: int = 0, name_prefix: str = 'rl-video', fps: int | None = None, disable_logger: bool = True, gc_trigger: Callable[[int], bool] | None = lambda episode: ...)

Adds support for video recording for Vector-based environments.

The class is the same as gymnasium.wrappers.rendering.RecordVideo, but
expects multiple frames when rendering the environment (one for each
environment of the VectorEnv). Frames are concatenated into one frame such
that its aspect ratio is as close as possible to the desired one.

Examples - Run 5 environments for 200 timesteps, and save the video every 5 episodes:
>>> import os
>>> import gymnasium as gym
>>> from gymnasium.wrappers.vector import RecordVideo
>>> envs = gym.make_vec(“CartPole-v1”, num_envs=5, render_mode=”rgb_array”)
>>> envs = RecordVideo(
…     envs,
…     video_folder=”save_videos_5envs”,
…     video_aspect_ratio=(1,1),
…     episode_trigger=lambda t: t % 5 == 0,
… )
>>> \_ = envs.reset(seed=123)
>>> \_ = envs.action_space.seed(123)
>>> for i in range(200):
…     actions = envs.action_space.sample()
…     \_ = envs.step(actions)
>>> envs.close()
>>> len(os.listdir(“save_videos_5envs”))
2

* **Parameters:**
  * **env** – The environment that will be wrapped
  * **video_folder** (*str*) – The folder where the recordings will be stored
  * **video_aspect_ratio** (*tuple*) – the desired aspect ratio of the video concatenating all environments. For example, (1, 1) means an
    aspect ratio of 1:1, while (16, 9) means 16:9.
  * **record_first_only** (*bool*) – if True, only the first environment is recorded.
  * **episode_trigger** – Function that accepts an integer and returns `True` iff a recording should be started at this episode
  * **step_trigger** – Function that accepts an integer and returns `True` iff a recording should be started at this step
  * **video_length** (*int*) – The length of recorded episodes. If 0, entire episodes are recorded.
    Otherwise, snippets of the specified length are captured
  * **name_prefix** (*str*) – Will be prepended to the filename of the recordings
  * **fps** (*int*) – The frame per second in the video. Provides a custom video fps for environment, if `None` then
    the environment metadata `render_fps` key is used if it exists, otherwise a default value of 30 is used.
  * **disable_logger** (*bool*) – Whether to disable moviepy logger or not, default it is disabled
  * **gc_trigger** – Function that accepts an integer and returns `True` iff garbage collection should be performed after this episode

#### NOTE
For vector environments that use same-step autoreset (see [https://farama.org/Vector-Autoreset-Mode](https://farama.org/Vector-Autoreset-Mode) for more details)
then the final frame of the episode will not be included in the video.

### *class* gymnasium.wrappers.vector.HumanRendering(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], screen_size: tuple[int, int] | None = None)

Adds support for Human-based Rendering for Vector-based environments.

* **Parameters:**
  * **env** – The vector environment
  * **screen_size** – The rendering screen size otherwise the environment sub-env render size is used

## Implemented observation wrappers

### *class* gymnasium.wrappers.vector.TransformObservation(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], func: Callable[[ObsType], WrapperObsType], observation_space: [Space](../spaces.md#gymnasium.spaces.Space) | None = None, single_observation_space: [Space](../spaces.md#gymnasium.spaces.Space) | None = None)

Transforms an observation via a function provided to the wrapper.

This function allows the manual specification of the vector-observation function as well as the single-observation function.
This is desirable when, for example, it is possible to process vector observations in parallel or via other more optimized methods.
Otherwise, the `VectorizeTransformObservation` should be used instead, where only `single_func` needs to be defined.

Example - Without observation transformation:
: ```pycon
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
  >>> obs, info = envs.reset(seed=123)
  >>> obs
  array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
         [ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598],
         [ 0.03517495, -0.000635  , -0.01098382, -0.03203924]],
        dtype=float32)
    >>> envs.close()
  ```

Example - With observation transformation:
: ```pycon
  >>> import gymnasium as gym
  >>> from gymnasium.spaces import Box
  >>> def scale_and_shift(obs):
  ...     return (obs - 1.0) * 2.0
  ...
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
  >>> new_obs_space = Box(low=envs.observation_space.low, high=envs.observation_space.high)
  >>> envs = TransformObservation(envs, func=scale_and_shift, observation_space=new_obs_space)
  >>> obs, info = envs.reset(seed=123)
  >>> obs
  array([[-1.9635296, -2.0892358, -2.055928 , -2.0631256],
         [-1.9429494, -1.9428282, -1.9061728, -1.9503881],
         [-1.9296501, -2.00127  , -2.0219676, -2.0640786]], dtype=float32)
  >>> envs.close()
  ```

* **Parameters:**
  * **env** – The vector environment to wrap
  * **func** – A function that will transform the vector observation. If this transformed observation is outside the observation space of `env.observation_space` then provide an `observation_space`.
  * **observation_space** – The observation spaces of the wrapper. If None, then it is computed from `single_observation_space`. If `single_observation_space` is not provided either, then it is assumed to be the same as `env.observation_space`.
  * **single_observation_space** – The observation space of the non-vectorized environment. If None, then it is assumed the same as `env.single_observation_space`.

### *class* gymnasium.wrappers.vector.FilterObservation(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], filter_keys: Sequence[str | int])

Vector wrapper for filtering dict or tuple observation spaces.

Example - Create a vectorized environment with a Dict space to demonstrate how to filter keys:
: ```pycon
  >>> import numpy as np
  >>> import gymnasium as gym
  >>> from gymnasium.spaces import Dict, Box
  >>> from gymnasium.wrappers import TransformObservation
  >>> from gymnasium.wrappers.vector import VectorizeTransformObservation, FilterObservation
  >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
  >>> make_dict = lambda x: {"obs": x, "junk": np.array([0.0])}
  >>> new_space = Dict({"obs": envs.single_observation_space, "junk": Box(low=-1.0, high=1.0)})
  >>> envs = VectorizeTransformObservation(env=envs, wrapper=TransformObservation, func=make_dict, observation_space=new_space)
  >>> envs = FilterObservation(envs, ["obs"])
  >>> obs, info = envs.reset(seed=123)
  >>> envs.close()
  >>> obs
  {'obs': array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
         [ 0.02852531,  0.02858594,  0.0469136 ,  0.02480598],
         [ 0.03517495, -0.000635  , -0.01098382, -0.03203924]],
        dtype=float32)}
  ```

* **Parameters:**
  * **env** – The vector environment to wrap
  * **filter_keys** – The subspaces to be included, use a list of strings or integers for `Dict` and `Tuple` spaces respectively

### *class* gymnasium.wrappers.vector.FlattenObservation(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType])

Observation wrapper that flattens the observation.

### Example

```pycon
>>> import gymnasium as gym
>>> envs = gym.make_vec("CarRacing-v3", num_envs=3, vectorization_mode="sync")
>>> obs, info = envs.reset(seed=123)
>>> obs.shape
(3, 96, 96, 3)
>>> envs = FlattenObservation(envs)
>>> obs, info = envs.reset(seed=123)
>>> obs.shape
(3, 27648)
>>> envs.close()
```

* **Parameters:**
  **env** – The vector environment to wrap

### *class* gymnasium.wrappers.vector.GrayscaleObservation(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], keep_dim: bool = False)

Observation wrapper that converts an RGB image to grayscale.

### Example

```pycon
>>> import gymnasium as gym
>>> envs = gym.make_vec("CarRacing-v3", num_envs=3, vectorization_mode="sync")
>>> obs, info = envs.reset(seed=123)
>>> obs.shape
(3, 96, 96, 3)
>>> envs = GrayscaleObservation(envs)
>>> obs, info = envs.reset(seed=123)
>>> obs.shape
(3, 96, 96)
>>> envs.close()
```

* **Parameters:**
  * **env** – The vector environment to wrap
  * **keep_dim** – If to keep the channel in the observation, if `True`, `obs.shape == 3` else `obs.shape == 2`

### *class* gymnasium.wrappers.vector.ResizeObservation(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], shape: tuple[int, ...])

Resizes image observations using OpenCV to shape.

### Example

```pycon
>>> import gymnasium as gym
>>> envs = gym.make_vec("CarRacing-v3", num_envs=3, vectorization_mode="sync")
>>> obs, info = envs.reset(seed=123)
>>> obs.shape
(3, 96, 96, 3)
>>> envs = ResizeObservation(envs, shape=(28, 28))
>>> obs, info = envs.reset(seed=123)
>>> obs.shape
(3, 28, 28, 3)
>>> envs.close()
```

* **Parameters:**
  * **env** – The vector environment to wrap
  * **shape** – The resized observation shape

### *class* gymnasium.wrappers.vector.ReshapeObservation(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], shape: int | tuple[int, ...])

Reshapes array based observations to shapes.

### Example

```pycon
>>> import gymnasium as gym
>>> envs = gym.make_vec("CarRacing-v3", num_envs=3, vectorization_mode="sync")
>>> obs, info = envs.reset(seed=123)
>>> obs.shape
(3, 96, 96, 3)
>>> envs = ReshapeObservation(envs, shape=(9216, 3))
>>> obs, info = envs.reset(seed=123)
>>> obs.shape
(3, 9216, 3)
>>> envs.close()
```

* **Parameters:**
  * **env** – The vector environment to wrap
  * **shape** – The reshaped observation space

### *class* gymnasium.wrappers.vector.RescaleObservation(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], min_obs: float | floating | integer | ndarray, max_obs: float | floating | integer | ndarray)

Linearly rescales observation to between a minimum and maximum value.

### Example

```pycon
>>> import gymnasium as gym
>>> envs = gym.make_vec("MountainCar-v0", num_envs=3, vectorization_mode="sync")
>>> obs, info = envs.reset(seed=123)
>>> obs.min()
np.float32(-0.46352962)
>>> obs.max()
np.float32(0.0)
>>> envs = RescaleObservation(envs, min_obs=-5.0, max_obs=5.0)
>>> obs, info = envs.reset(seed=123)
>>> obs.min()
np.float32(-0.90849805)
>>> obs.max()
np.float32(0.0)
>>> envs.close()
```

* **Parameters:**
  * **env** – The vector environment to wrap
  * **min_obs** – The new minimum observation bound
  * **max_obs** – The new maximum observation bound

### *class* gymnasium.wrappers.vector.DtypeObservation(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], dtype: Any)

Observation wrapper for transforming the dtype of an observation.

### Example

```pycon
>>> import numpy as np
>>> import gymnasium as gym
>>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
>>> obs, info = envs.reset(seed=123)
>>> obs.dtype
dtype('float32')
>>> envs = DtypeObservation(envs, dtype=np.float64)
>>> obs, info = envs.reset(seed=123)
>>> obs.dtype
dtype('float64')
>>> envs.close()
```

* **Parameters:**
  * **env** – The vector environment to wrap
  * **dtype** – The new dtype of the observation

### *class* gymnasium.wrappers.vector.NormalizeObservation(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ndarray, ActType, ArrayType], epsilon: float = 1e-8)

This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

The property \_update_running_mean allows to freeze/continue the running mean calculation of the observation
statistics. If True (default), the RunningMeanStd will get updated every step and reset call.
If False, the calculated statistics are used but not updated anymore; this may be used during evaluation.

#### NOTE
The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
newly instantiated or the policy was changed recently.

#### NOTE
For same-step autoreset, `info["final_obs"]` is normalized and included in the statistics before the
step’s observations, matching the update order of [`gymnasium.wrappers.NormalizeObservation`](../wrappers/observation_wrappers.md#gymnasium.wrappers.NormalizeObservation).
Partial resets (`reset_mask` with any `False` values) are not supported for disabled autoreset.

Example without the normalize observation wrapper:
: ```pycon
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
  >>> obs, info = envs.reset(seed=123)
  >>> _ = envs.action_space.seed(123)
  >>> for _ in range(100):
  ...     obs, *_ = envs.step(envs.action_space.sample())
  >>> np.mean(obs)
  np.float32(0.024251968)
  >>> np.std(obs)
  np.float32(0.62259156)
  >>> envs.close()
  ```

Example with the normalize observation wrapper:
: ```pycon
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
  >>> envs = NormalizeObservation(envs)
  >>> obs, info = envs.reset(seed=123)
  >>> _ = envs.action_space.seed(123)
  >>> for _ in range(100):
  ...     obs, *_ = envs.step(envs.action_space.sample())
  >>> np.mean(obs)
  np.float32(-0.2359734)
  >>> np.std(obs)
  np.float32(1.1938739)
  >>> envs.close()
  ```

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The environment to apply the wrapper
  * **epsilon** – A stability parameter that is used when scaling the observations.
* **Raises:**
  [**InvalidBound**](../utils.md#gymnasium.error.InvalidBound) – If `epsilon` is not finite and strictly positive.

## Implemented action wrappers

### *class* gymnasium.wrappers.vector.TransformAction(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], func: Callable[[WrapperActType], ActType], action_space: [Space](../spaces.md#gymnasium.spaces.Space) | None = None, single_action_space: [Space](../spaces.md#gymnasium.spaces.Space) | None = None)

Transforms an action via a function provided to the wrapper.

The function `func` will be applied to all vector actions.
If the observations from `func` are outside the bounds of the `env`’s action space,
provide an `action_space` which specifies the action space for the vectorized environment.

Example - Without action transformation:
: ```pycon
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
  >>> _ = envs.action_space.seed(123)
  >>> obs, info = envs.reset(seed=123)
  >>> for _ in range(10):
  ...     obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
  ...
  >>> envs.close()
  >>> obs
  array([[-0.46553135, -0.00142543],
         [-0.498371  , -0.00715587],
         [-0.46515748, -0.00624371]], dtype=float32)
  ```

Example - With action transformation:
: ```pycon
  >>> import gymnasium as gym
  >>> from gymnasium.spaces import Box
  >>> def shrink_action(act):
  ...     return act * 0.3
  ...
  >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
  >>> new_action_space = Box(low=shrink_action(envs.action_space.low), high=shrink_action(envs.action_space.high))
  >>> envs = TransformAction(env=envs, func=shrink_action, action_space=new_action_space)
  >>> _ = envs.action_space.seed(123)
  >>> obs, info = envs.reset(seed=123)
  >>> for _ in range(10):
  ...     obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
  ...
  >>> envs.close()
  >>> obs
  array([[-0.48468155, -0.00372536],
         [-0.47599354, -0.00545912],
         [-0.46543318, -0.00615723]], dtype=float32)
  ```

* **Parameters:**
  * **env** – The vector environment to wrap
  * **func** – A function that will transform an action. If this transformed action is outside the action space of `env.action_space` then provide an `action_space`.
  * **action_space** – The action spaces of the wrapper. If None, then it is computed from `single_action_space`. If `single_action_space` is not provided either, then it is assumed to be the same as `env.action_space`.
  * **single_action_space** – The action space of the non-vectorized environment. If None, then it is assumed the same as `env.single_action_space`.

### *class* gymnasium.wrappers.vector.ClipAction(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType])

Clip the continuous action within the valid `Box` observation space bound.

Example - Passing an out-of-bounds action to the environment to be clipped.
: ```pycon
  >>> import numpy as np
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
  >>> envs = ClipAction(envs)
  >>> _ = envs.action_space.seed(123)
  >>> obs, info = envs.reset(seed=123)
  >>> obs, rew, term, trunc, info = envs.step(np.array([5.0, -5.0, 2.0]))
  >>> envs.close()
  >>> obs
  array([[-0.4624777 ,  0.00105192],
         [-0.44504836, -0.00209899],
         [-0.42884544,  0.00080468]], dtype=float32)
  ```

* **Parameters:**
  **env** – The vector environment to wrap

### *class* gymnasium.wrappers.vector.RescaleAction(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], min_action: float | int | ndarray, max_action: float | int | ndarray)

Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

Example - Without action scaling:
: ```pycon
  >>> import numpy as np
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
  >>> _ = envs.action_space.seed(123)
  >>> obs, info = envs.reset(seed=123)
  >>> for _ in range(10):
  ...     obs, rew, term, trunc, info = envs.step(0.5 * np.ones((3, 1)))
  ...
  >>> envs.close()
  >>> obs
  array([[-0.44799727,  0.00266526],
         [-0.4351738 ,  0.00133522],
         [-0.42683297,  0.00048403]], dtype=float32)
  ```

Example - With action scaling:
: ```pycon
  >>> import numpy as np
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
  >>> envs = RescaleAction(envs, 0.0, 1.0)
  >>> _ = envs.action_space.seed(123)
  >>> obs, info = envs.reset(seed=123)
  >>> for _ in range(10):
  ...     obs, rew, term, trunc, info = envs.step(0.5 * np.ones((3, 1)))
  ...
  >>> envs.close()
  >>> obs
  array([[-0.48657528, -0.00395268],
         [-0.47377947, -0.00529102],
         [-0.46546045, -0.00614867]], dtype=float32)
  ```

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The vector environment to wrap
  * **min_action** (*float* *,* *int* *or* *np.ndarray*) – The min values for each action. This may be a numpy array or a scalar.
  * **max_action** (*float* *,* *int* *or* *np.ndarray*) – The max values for each action. This may be a numpy array or a scalar.

## Implemented reward wrappers

### *class* gymnasium.wrappers.vector.TransformReward(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ArrayType], func: Callable[[ArrayType], ArrayType])

A reward wrapper that allows a custom function to modify the step reward.

Example with reward transformation:
: ```pycon
  >>> import gymnasium as gym
  >>> from gymnasium.spaces import Box
  >>> def scale_and_shift(rew):
  ...     return (rew - 1.0) * 2.0
  ...
  >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
  >>> envs = TransformReward(env=envs, func=scale_and_shift)
  >>> _ = envs.action_space.seed(123)
  >>> obs, info = envs.reset(seed=123)
  >>> obs, rew, term, trunc, info = envs.step(envs.action_space.sample())
  >>> envs.close()
  >>> obs
  array([[-4.6343064e-01,  9.8971417e-05],
         [-4.4488689e-01, -1.9375233e-03],
         [-4.3118435e-01, -1.5342437e-03]], dtype=float32)
  ```

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The vector environment to wrap
  * **func** – (Callable): The function to apply to reward

### *class* gymnasium.wrappers.vector.ClipReward(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, \_ArrayT], min_reward: float | ndarray | None = None, max_reward: float | ndarray | None = None)

A wrapper that clips the rewards for an environment between an upper and lower bound.

Example with clipped rewards:
: ```pycon
  >>> import numpy as np
  >>> import gymnasium as gym
  >>> envs = gym.make_vec("MountainCarContinuous-v0", num_envs=3)
  >>> envs = ClipReward(envs, 0.0, 2.0)
  >>> _ = envs.action_space.seed(123)
  >>> obs, info = envs.reset(seed=123)
  >>> for _ in range(10):
  ...     obs, rew, term, trunc, info = envs.step(0.5 * np.ones((3, 1)))
  ...
  >>> envs.close()
  >>> rew
  array([0., 0., 0.])
  ```

* **Parameters:**
  * **env** – The vector environment to wrap
  * **min_reward** – The min reward for each step
  * **max_reward** – the max reward for each step

### *class* gymnasium.wrappers.vector.NormalizeReward(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, ndarray], gamma: float = 0.99, epsilon: float = 1e-8)

This wrapper will scale rewards s.t. their exponential moving average has an approximately fixed variance.

The property \_update_running_mean allows to freeze/continue the running mean calculation of the reward
statistics. If True (default), the RunningMeanStd will get updated every time self.normalize() is called.
If False, the calculated statistics are used but not updated anymore; this may be used during evaluation.

#### NOTE
The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
instantiated or the policy was changed recently.

#### NOTE
Unlike [`gymnasium.wrappers.NormalizeReward`](../wrappers/reward_wrappers.md#gymnasium.wrappers.NormalizeReward), which carries the discounted return into the next episode
unless the episode terminates, the accumulated return of a sub-environment is restarted once it is reset,
i.e., at the end of an episode for same-step autoreset and on `reset` for disabled autoreset.

Example without the normalize reward wrapper:
: ```pycon
  >>> import gymnasium as gym
  >>> import numpy as np
  >>> envs = gym.make_vec("MountainCarContinuous-v0", 3)
  >>> _ = envs.reset(seed=123)
  >>> _ = envs.action_space.seed(123)
  >>> episode_rewards = []
  >>> for _ in range(100):
  ...     observation, reward, *_ = envs.step(envs.action_space.sample())
  ...     episode_rewards.append(reward)
  ...
  >>> envs.close()
  >>> np.mean(episode_rewards)
  np.float64(-0.03359492141887935)
  >>> np.std(episode_rewards)
  np.float64(0.029028230434438706)
  ```

Example with the normalize reward wrapper:
: ```pycon
  >>> import gymnasium as gym
  >>> import numpy as np
  >>> envs = gym.make_vec("MountainCarContinuous-v0", 3)
  >>> envs = NormalizeReward(envs)
  >>> _ = envs.reset(seed=123)
  >>> _ = envs.action_space.seed(123)
  >>> episode_rewards = []
  >>> for _ in range(100):
  ...     observation, reward, *_ = envs.step(envs.action_space.sample())
  ...     episode_rewards.append(reward)
  ...
  >>> envs.close()
  >>> np.mean(episode_rewards)
  np.float64(-0.1598639409552856)
  >>> np.std(episode_rewards)
  np.float64(0.2780030923586878)
  ```

* **Parameters:**
  * **env** (*env*) – The environment to apply the wrapper
  * **epsilon** (*float*) – A stability parameter
  * **gamma** (*float*) – The discount factor that is used in the exponential moving average.
* **Raises:**
  [**InvalidBound**](../utils.md#gymnasium.error.InvalidBound) – If `gamma` is outside `[0, 1]`, or if `epsilon` is not finite and strictly positive.

## Implemented data conversion wrappers

### *class* gymnasium.wrappers.vector.ArrayConversion(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv)[ObsType, ActType, Any], env_xp: ModuleType, target_xp: ModuleType, env_device: Any | None = None, target_device: Any | None = None)

Wraps a vector environment returning Array API compatible arrays so that it can be interacted with through a specific framework.

Popular Array API frameworks include `numpy`, `torch`, `jax.numpy`, `cupy` etc. With this wrapper, you can convert outputs from your environment to
any of these frameworks. Conversely, actions are automatically mapped back to the environment framework, if possible without moving the
data or device transfers.

### Notes

A vectorized version of [`gymnasium.wrappers.ArrayConversion`](../wrappers/misc_wrappers.md#gymnasium.wrappers.ArrayConversion)

### Example

```pycon
>>> import gymnasium as gym                                         
>>> envs = gym.make_vec("JaxEnv-vx", 3)                             
>>> envs = ArrayConversion(envs, xp=np)                             
```

* **Parameters:**
  * **env** – The Array API compatible environment to wrap
  * **env_xp** – The Array API framework the environment is on
  * **target_xp** – The Array API framework to convert to
  * **env_device** – The device the environment is on
  * **target_device** – The device on which Arrays should be returned

### *class* gymnasium.wrappers.vector.JaxToNumpy(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv))

Wraps a jax vector environment so that it can be interacted with through numpy arrays.

### Notes

A vectorized version of [`gymnasium.wrappers.JaxToNumpy`](../wrappers/misc_wrappers.md#gymnasium.wrappers.JaxToNumpy)

Actions must be provided as numpy arrays and observations, rewards, terminations and truncations will be returned as numpy arrays.

### Example

```pycon
>>> import gymnasium as gym                                         
>>> envs = gym.make_vec("JaxEnv-vx", 3)                             
>>> envs = JaxToNumpy(envs)                                         
```

* **Parameters:**
  **env** – the vector jax environment to wrap

### *class* gymnasium.wrappers.vector.JaxToTorch(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv), device: str | device | None = None)

Wraps a Jax-based vector environment so that it can be interacted with through PyTorch Tensors.

Actions must be provided as PyTorch Tensors and observations, rewards, terminations and truncations will be returned as PyTorch Tensors.

### Example

```pycon
>>> import gymnasium as gym                                         
>>> envs = gym.make_vec("JaxEnv-vx", 3)                             
>>> envs = JaxToTorch(envs)                                         
```

* **Parameters:**
  * **env** – The Jax-based vector environment to wrap
  * **device** – The device the torch Tensors should be moved to

### *class* gymnasium.wrappers.vector.NumpyToTorch(env: [VectorEnv](../vector.md#gymnasium.vector.VectorEnv), device: str | device | None = None)

Wraps a numpy-based environment so that it can be interacted with through PyTorch Tensors.

### Example

```pycon
>>> import torch
>>> import gymnasium as gym
>>> from gymnasium.wrappers.vector import NumpyToTorch
>>> envs = gym.make_vec("CartPole-v1", 3)
>>> envs = NumpyToTorch(envs)
>>> obs, _ = envs.reset(seed=123)
>>> type(obs)
<class 'torch.Tensor'>
>>> action = torch.tensor(envs.action_space.sample())
>>> obs, reward, terminated, truncated, info = envs.step(action)
>>> envs.close()
>>> type(obs)
<class 'torch.Tensor'>
>>> type(reward)
<class 'torch.Tensor'>
>>> type(terminated)
<class 'torch.Tensor'>
>>> type(truncated)
<class 'torch.Tensor'>
```

* **Parameters:**
  * **env** – The NumPy-based vector environment to wrap
  * **device** – The device the torch Tensors should be moved to
