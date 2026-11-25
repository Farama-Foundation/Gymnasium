# Misc Wrappers

## Common wrappers

### *class* gymnasium.wrappers.TimeLimit(env: [Env](../env.md#gymnasium.Env), max_episode_steps: int)

Limits the number of steps for an environment through truncating the environment if a maximum number of timesteps is exceeded.

If a truncation is not defined inside the environment itself, this is the only place that the truncation signal is issued.
Critically, this is different from the terminated signal that originates from the underlying environment as part of the MDP.
No vector wrapper exists.

Example using the TimeLimit wrapper:
: ```pycon
  >>> from gymnasium.wrappers import TimeLimit
  >>> from gymnasium.envs.classic_control import CartPoleEnv
  ```
  <br/>
  ```pycon
  >>> spec = gym.spec("CartPole-v1")
  >>> spec.max_episode_steps
  500
  >>> env = gym.make("CartPole-v1")
  >>> env  # TimeLimit is included within the environment stack
  <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
  >>> env.spec  
  EnvSpec(id='CartPole-v1', ..., max_episode_steps=500, ...)
  >>> env = gym.make("CartPole-v1", max_episode_steps=3)
  >>> env.spec  
  EnvSpec(id='CartPole-v1', ..., max_episode_steps=3, ...)
  >>> env = TimeLimit(CartPoleEnv(), max_episode_steps=10)
  >>> env
  <TimeLimit<CartPoleEnv instance>>
  ```

Example of TimeLimit determining the episode step
: ```pycon
  >>> env = gym.make("CartPole-v1", max_episode_steps=3)
  >>> _ = env.reset(seed=123)
  >>> _ = env.action_space.seed(123)
  >>> _, _, terminated, truncated, _ = env.step(env.action_space.sample())
  >>> terminated, truncated
  (False, False)
  >>> _, _, terminated, truncated, _ = env.step(env.action_space.sample())
  >>> terminated, truncated
  (False, False)
  >>> _, _, terminated, truncated, _ = env.step(env.action_space.sample())
  >>> terminated, truncated
  (False, True)
  ```

Change logs:
: * v0.10.6 - Initially added
  * v0.25.0 - With the step API update, the termination and truncation signal is returned separately.

* **Parameters:**
  * **env** – The environment to apply the wrapper
  * **max_episode_steps** – the environment step after which the episode is truncated (`elapsed >= max_episode_steps`)

### *class* gymnasium.wrappers.RecordVideo(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], video_folder: str, episode_trigger: Callable[[int], bool] | None = None, step_trigger: Callable[[int], bool] | None = None, video_length: int = 0, name_prefix: str = 'rl-video', fps: int | None = None, disable_logger: bool = True, gc_trigger: Callable[[int], bool] | None = lambda episode: ...)

Records videos of environment episodes using the environment’s render function.

Usually, you only want to record episodes intermittently, say every hundredth episode or at every thousandth environment step.
To do this, you can specify `episode_trigger` or `step_trigger`.
They should be functions returning a boolean that indicates whether a recording should be started at the
current episode or step, respectively.

The `episode_trigger` should return `True` on the episode when recording should start.
The `step_trigger` should return `True` on the n-th environment step that the recording should be started, where n sums over all previous episodes.
If neither `episode_trigger` nor `step_trigger` is passed, a default `episode_trigger` will be employed, i.e. [`capped_cubic_video_schedule()`](../utils.md#gymnasium.utils.save_video.capped_cubic_video_schedule).
This function starts a video at every episode that is a power of 3 until 1000 and then every 1000 episodes.
By default, the recording will be stopped once reset is called.
However, you can also create recordings of fixed length (possibly spanning several episodes)
by passing a strictly positive value for `video_length`.

Examples - Run the environment for 50 episodes, and save the video every 10 episodes starting from the 0th:
: ```pycon
  >>> import os
  >>> import gymnasium as gym
  >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
  >>> trigger = lambda t: t % 10 == 0
  >>> env = RecordVideo(env, video_folder="./save_videos1", episode_trigger=trigger, disable_logger=True)
  >>> for i in range(50):
  ...     termination, truncation = False, False
  ...     _ = env.reset(seed=123)
  ...     while not (termination or truncation):
  ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
  ...
  >>> env.close()
  >>> len(os.listdir("./save_videos1"))
  5
  ```

Examples - Run the environment for 5 episodes, start a recording every 200th step, making sure each video is 100 frames long:
: ```pycon
  >>> import os
  >>> import gymnasium as gym
  >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
  >>> trigger = lambda t: t % 200 == 0
  >>> env = RecordVideo(env, video_folder="./save_videos2", step_trigger=trigger, video_length=100, disable_logger=True)
  >>> for i in range(5):
  ...     termination, truncation = False, False
  ...     _ = env.reset(seed=123)
  ...     _ = env.action_space.seed(123)
  ...     while not (termination or truncation):
  ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
  ...
  >>> env.close()
  >>> len(os.listdir("./save_videos2"))
  2
  ```

Examples - Run 3 episodes, record everything, but in chunks of 1000 frames:
: ```pycon
  >>> import os
  >>> import gymnasium as gym
  >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
  >>> env = RecordVideo(env, video_folder="./save_videos3", video_length=1000, disable_logger=True)
  >>> for i in range(3):
  ...     termination, truncation = False, False
  ...     _ = env.reset(seed=123)
  ...     while not (termination or truncation):
  ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
  ...
  >>> env.close()
  >>> len(os.listdir("./save_videos3"))
  2
  ```

Change logs:
: * v0.25.0 - Initially added to replace `wrappers.monitoring.VideoRecorder`

* **Parameters:**
  * **env** – The environment that will be wrapped
  * **video_folder** (*str*) – The folder where the recordings will be stored
  * **episode_trigger** – Function that accepts an integer and returns `True` iff a recording should be started at this episode
  * **step_trigger** – Function that accepts an integer and returns `True` iff a recording should be started at this step
  * **video_length** (*int*) – The length of recorded episodes. If 0, entire episodes are recorded.
    Otherwise, snippets of the specified length are captured
  * **name_prefix** (*str*) – Will be prepended to the filename of the recordings
  * **fps** (*int*) – The frame per second in the video. Provides a custom video fps for environment, if `None` then
    the environment metadata `render_fps` key is used if it exists, otherwise a default value of 30 is used.
  * **disable_logger** (*bool*) – Whether to disable moviepy logger or not, default it is disabled
  * **gc_trigger** – Function that accepts an integer and returns `True` iff garbage collection should be performed after this episode

### *class* gymnasium.wrappers.RecordEpisodeStatistics(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], buffer_length: int = 100, stats_key: str = 'episode')

This wrapper will keep track of cumulative rewards and episode lengths.

At the end of an episode, the statistics of the episode will be added to `info`
using the key `episode`. If using a vectorized environment also the key
`_episode` is used which indicates whether the env at the respective index has
the episode statistics.
A vector version of the wrapper exists, [`gymnasium.wrappers.vector.RecordEpisodeStatistics`](../vector/wrappers.md#gymnasium.wrappers.vector.RecordEpisodeStatistics).

After the completion of an episode, `info` will look like this:

```default
>>> info = {
...     "episode": {
...         "r": "<cumulative reward>",
...         "l": "<episode length>",
...         "t": "<elapsed time since beginning of episode>"
...     },
... }
```

For a vectorized environments the output will be in the form of:

```default
>>> infos = {
...     "episode": {
...         "r": "<array of cumulative reward>",
...         "l": "<array of episode length>",
...         "t": "<array of elapsed time since beginning of episode>"
...     },
...     "_episode": "<boolean array of length num-envs>"
... }
```

Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
`wrapped_env.return_queue` and `wrapped_env.length_queue` respectively.

The statistics restart for each episode depending on `env.metadata["autoreset_mode"]`:
: * `AutoresetMode.NEXT_STEP` (see [`Autoreset`](#gymnasium.wrappers.Autoreset)) - The step after an episode ends resets the environment
    so isn’t counted in the next episode.
  * `AutoresetMode.SAME_STEP` - The environment resets within the step that the episode ends,
    so the statistics restart immediately.
  * `AutoresetMode.DISABLED` or no `autoreset_mode` - The statistics restart when `reset()` is called.

* **Variables:**
  * **time_queue** ( *\**) – The time length of the last `deque_size`-many episodes
  * **return_queue** ( *\**) – The cumulative rewards of the last `deque_size`-many episodes
  * **length_queue** ( *\**) – The lengths of the last `deque_size`-many episodes

Change logs:
: * v0.15.4 - Initially added
  * v1.0.0 - Removed vector environment support (see [`gymnasium.wrappers.vector.RecordEpisodeStatistics`](../vector/wrappers.md#gymnasium.wrappers.vector.RecordEpisodeStatistics)) and add attribute `time_queue`
  * v1.5.0 - Fixed the statistics accumulating across episodes for autoreset environments

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The environment to apply the wrapper
  * **buffer_length** – The size of the buffers `return_queue`, `length_queue` and `time_queue`
  * **stats_key** – The info key for the episode statistics

### *class* gymnasium.wrappers.AtariPreprocessing(env: [Env](../env.md#gymnasium.Env), noop_max: int = 30, frame_skip: int = 4, screen_size: int | tuple[int, int] = 84, terminal_on_life_loss: bool = False, grayscale_obs: bool = True, grayscale_newaxis: bool = False, scale_obs: bool = False)

Implements the common preprocessing techniques for Atari environments (excluding frame stacking).

For frame stacking use [`gymnasium.wrappers.FrameStackObservation`](observation_wrappers.md#gymnasium.wrappers.FrameStackObservation).
No vector version of the wrapper exists

This class follows the guidelines in Machado et al. (2018),
“Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents”.

Specifically, the following preprocess stages applies to the atari environment:

- Noop Reset: Obtains the initial state by taking a random number of no-ops on reset, default max 30 no-ops.
- Frame skipping: The number of frames skipped between steps, 4 by default.
- Max-pooling: Pools over the most recent two observations from the frame skips.
- Termination signal when a life is lost: When the agent losses a life during the environment, then the environment is terminated.
  : Turned off by default. Not recommended by Machado et al. (2018).
- Resize to a square image: Resizes the atari environment original observation shape from 210x180 to 84x84 by default.
- Grayscale observation: Makes the observation greyscale, enabled by default.
- Grayscale new axis: Extends the last channel of the observation such that the image is 3-dimensional, not enabled by default.
- Scale observation: Whether to scale the observation between [0, 1) or [0, 255), not scaled by default.

### Example

```pycon
>>> import gymnasium as gym
>>> import ale_py
>>> gym.register_envs(ale_py)
>>> env = gym.make("ALE/Pong-v5", frameskip=1)
>>> env = AtariPreprocessing(
...     env,
...     noop_max=10, frame_skip=4, terminal_on_life_loss=True,
...     screen_size=84, grayscale_obs=False, grayscale_newaxis=False
... )
```

Change logs:
: * Added in gym v0.12.2 (gym #1455)

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The environment to apply the preprocessing
  * **noop_max** (*int*) – For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
  * **frame_skip** (*int*) – The number of frames between new observation the agents observations effecting the frequency at which the agent experiences the game.
  * **screen_size** (*int* *|* *tuple* *[**int* *,* *int* *]*) – resize Atari frame.
  * **terminal_on_life_loss** (*bool*) – if True, then `step()` returns terminated=True whenever a
    life is lost.
  * **grayscale_obs** (*bool*) – if True, then gray scale observation is returned, otherwise, RGB observation
    is returned.
  * **grayscale_newaxis** (*bool*) – if True and grayscale_obs=True, then a channel axis is added to
    grayscale observations to make them 3-dimensional.
  * **scale_obs** (*bool*) – if True, then observation normalized in range [0,1) is returned. It also limits memory
    optimization benefits of FrameStack Wrapper.
* **Raises:**
  * [**DependencyNotInstalled**](../utils.md#gymnasium.error.DependencyNotInstalled) – opencv-python package not installed
  * **ValueError** – Disable frame-skipping in the original env

## Uncommon wrappers

### *class* gymnasium.wrappers.Autoreset(env: [Env](../env.md#gymnasium.Env), autoreset_mode: str | [AutoresetMode](../vector/autoreset.md#gymnasium.vector.AutoresetMode) = AutoresetMode.NEXT_STEP)

The wrapped environment is automatically reset when a terminated or truncated state is reached.

This follows the vector autoreset api (see [https://gymnasium.farama.org/api/vector/autoreset/](https://gymnasium.farama.org/api/vector/autoreset/)) with two modes:
: * `AutoresetMode.NEXT_STEP` - On the step after an episode terminates or truncates, the environment is reset.
  * `AutoresetMode.SAME_STEP` - On the step that an episode terminates or truncates, the environment is reset,
    returning the reset observation and info with the episode’s final observation and info in
    `info["final_obs"]` and `info["final_info"]`.

The autoreset mode is added to the environment metadata as `"autoreset_mode"`.

Change logs:
: * v0.24.0 - Initially added as AutoResetWrapper
  * v1.0.0 - renamed to Autoreset and autoreset order was changed to reset on the step after the environment terminates or truncates. As a result, “final_observation” and “final_info” is removed.
  * v1.5.0 - Adds `autoreset_mode` argument for `AutoresetMode.SAME_STEP` support and `"autoreset_mode"` to the environment metadata

* **Parameters:**
  * **env** (*gym.Env*) – The environment to apply the wrapper
  * **autoreset_mode** – The autoreset mode, either `AutoresetMode.NEXT_STEP` or `AutoresetMode.SAME_STEP`

### *class* gymnasium.wrappers.PassiveEnvChecker(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType])

A passive wrapper that surrounds the `step`, `reset` and `render` functions to check they follow Gymnasium’s API.

This wrapper is automatically applied during make and can be disabled with disable_env_checker.
No vector version of the wrapper exists.

### Example

```pycon
>>> import gymnasium as gym
>>> env = gym.make("CartPole-v1")
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
>>> env = gym.make("CartPole-v1", disable_env_checker=True)
>>> env
<TimeLimit<OrderEnforcing<CartPoleEnv<CartPole-v1>>>>
```

Change logs:
: * v0.24.1 - Initially added however broken in several ways
  * v0.25.0 - Bugs was all fixed
  * v0.29.0 - Removed warnings for infinite bounds for Box observation and action spaces and irregular bound shapes

Initialises the wrapper with the environments, run the observation and action space tests.

### *class* gymnasium.wrappers.HumanRendering(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType])

Allows human like rendering for environments that support “rgb_array” rendering.

This wrapper is particularly useful when you have implemented an environment that can produce
RGB images but haven’t implemented any code to render the images to the screen.
If you want to use this wrapper with your environments, remember to specify `"render_fps"`
in the metadata of your environment.

The `render_mode` of the wrapped environment must be either `'rgb_array'` or `'rgb_array_list'`.

No vector version of the wrapper exists.

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import HumanRendering
>>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
>>> wrapped = HumanRendering(env)
>>> obs, _ = wrapped.reset()     # This will start rendering to the screen
```

The wrapper can also be applied directly when the environment is instantiated, simply by passing
`render_mode="human"` to `make`. The wrapper will only be applied if the environment does not
implement human-rendering natively (i.e. `render_mode` does not contain `"human"`).

```pycon
>>> env = gym.make("phys2d/CartPole-v1", render_mode="human")      # CartPoleJax-v1 doesn't implement human-rendering natively
>>> obs, _ = env.reset()     # This will start rendering to the screen
```

Warning: If the base environment uses `render_mode="rgb_array_list"`, its (i.e. the *base environment’s*) render method
will always return an empty list:

```pycon
>>> env = gym.make("LunarLander-v3", render_mode="rgb_array_list")
>>> wrapped = HumanRendering(env)
>>> obs, _ = wrapped.reset()
>>> env.render() # env.render() will always return an empty list!
[]
```

Change logs:
: * v0.25.0 - Initially added

* **Parameters:**
  **env** – The environment that is being wrapped

### *class* gymnasium.wrappers.OrderEnforcing(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], disable_render_order_enforcing: bool = False)

Will produce an error if `step` or `render` is called before `reset`.

No vector version of the wrapper exists.

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import OrderEnforcing
>>> env = gym.make("CartPole-v1", render_mode="human")
>>> env = OrderEnforcing(env)
>>> env.step(0)
Traceback (most recent call last):
    ...
gymnasium.error.ResetNeeded: Cannot call env.step() before calling env.reset()
>>> env.render()
Traceback (most recent call last):
    ...
gymnasium.error.ResetNeeded: Cannot call `env.render()` before calling `env.reset()`, if this is an intended action, set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper.
>>> _ = env.reset()
>>> env.render()
>>> _ = env.step(0)
>>> env.close()
```

Change logs:
: * v0.22.0 - Initially added
  * v0.24.0 - Added order enforcing for the render function

* **Parameters:**
  * **env** – The environment to wrap
  * **disable_render_order_enforcing** – If to disable render order enforcing

### *class* gymnasium.wrappers.RenderCollection(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], pop_frames: bool = True, reset_clean: bool = True)

Collect rendered frames of an environment such `render` returns a `list[RenderedFrame]`.

No vector version of the wrapper exists.

Example - Return the list of frames for the number of steps `render` wasn’t called.
: ```pycon
  >>> import gymnasium as gym
  >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
  >>> env = RenderCollection(env)
  >>> _ = env.reset(seed=123)
  >>> for _ in range(5):
  ...     _ = env.step(env.action_space.sample())
  ...
  >>> frames = env.render()
  >>> len(frames)
  6
  ```
  <br/>
  ```pycon
  >>> frames = env.render()
  >>> len(frames)
  0
  ```

Example - Return the list of frames for the number of steps the episode was running.
: ```pycon
  >>> import gymnasium as gym
  >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
  >>> env = RenderCollection(env, pop_frames=False)
  >>> _ = env.reset(seed=123)
  >>> for _ in range(5):
  ...     _ = env.step(env.action_space.sample())
  ...
  >>> frames = env.render()
  >>> len(frames)
  6
  ```
  <br/>
  ```pycon
  >>> frames = env.render()
  >>> len(frames)
  6
  ```

Example - Collect all frames for all episodes, without clearing them when render is called
: ```pycon
  >>> import gymnasium as gym
  >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
  >>> env = RenderCollection(env, pop_frames=False, reset_clean=False)
  >>> _ = env.reset(seed=123)
  >>> for _ in range(5):
  ...     _ = env.step(env.action_space.sample())
  ...
  >>> _ = env.reset(seed=123)
  >>> for _ in range(5):
  ...     _ = env.step(env.action_space.sample())
  ...
  >>> frames = env.render()
  >>> len(frames)
  12
  ```
  <br/>
  ```pycon
  >>> frames = env.render()
  >>> len(frames)
  12
  ```

Change logs:
: * v0.26.2 - Initially added

* **Parameters:**
  * **env** – The environment that is being wrapped
  * **pop_frames** (*bool*) – If true, clear the collection frames after `meth:render` is called. Default value is `True`.
  * **reset_clean** (*bool*) – If true, clear the collection frames when `meth:reset` is called. Default value is `True`.

### *class* gymnasium.wrappers.AddWhiteNoise(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], probability_of_noise_per_pixel: float, is_noise_grayscale: bool = False)

Randomly replaces pixels with white noise.

If used with `render_mode="rgb_array"` and `AddRenderObservation`, it will
make observations noisy.
The environment may also become partially-observable, turning the MDP into a POMDP.

Example - Every pixel will be replaced by white noise with probability 0.5:
: ```pycon
  >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
  >>> env = AddWhiteNoise(env, probability_of_noise_per_pixel=0.5)
  >>> env = HumanRendering(env)
  >>> obs, _ = env.reset(seed=123)
  >>> obs, *_ = env.step(env.action_space.sample())
  ```

* **Parameters:**
  * **env** – The environment that is being wrapped
  * **probability_of_noise_per_pixel** – the probability that a pixel is white noise
  * **is_noise_grayscale** – if True, RGB noise is converted to grayscale

### *class* gymnasium.wrappers.ObstructView(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], obstructed_pixels_ratio: float, obstruction_width: int, is_noise_grayscale: bool = False)

Randomly obstructs rendering with white noise patches.

If used with `render_mode="rgb_array"` and `AddRenderObservation`, it will
make observations noisy.
The number of patches depends on how many pixels we want to obstruct.
Depending on the size of the patches, the environment may become
partially-observable, turning the MDP into a POMDP.

Example - Obstruct 50% of the pixels with patches of size 50x50 pixels:
: ```pycon
  >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
  >>> env = ObstructView(env, obstructed_pixels_ratio=0.5, obstruction_width=50)
  >>> env = HumanRendering(env)
  >>> obs, _ = env.reset(seed=123)
  >>> obs, *_ = env.step(env.action_space.sample())
  ```

* **Parameters:**
  * **env** – The environment that is being wrapped
  * **obstructed_pixels_ratio** – the percentage of pixels obstructed with white noise
  * **obstruction_width** – the width of the obstruction patches
  * **is_noise_grayscale** – if True, RGB noise is converted to grayscale

## Data conversion wrappers

### *class* gymnasium.wrappers.ArrayConversion(env: [Env](../env.md#gymnasium.Env), env_xp: ModuleType, target_xp: ModuleType, env_device: Any | None = None, target_device: Any | None = None)

Wraps an Array API compatible environment so that it can be interacted with another Array API framework.

Popular Array API frameworks include `numpy`, `torch`, `jax.numpy`, `cupy` etc. With this wrapper, you can convert outputs from your environment to
any of these frameworks. Conversely, actions are automatically mapped back to the environment framework, if possible without moving the
data or device transfers.

A vector version of the wrapper exists, [`gymnasium.wrappers.vector.ArrayConversion`](../vector/wrappers.md#gymnasium.wrappers.vector.ArrayConversion).

### Example

```pycon
>>> import torch                                                
>>> import jax.numpy as jnp                                     
>>> import gymnasium as gym                                     
>>> env = gym.make("JaxEnv-vx")                                 
>>> env = ArrayConversion(env, env_xp=jnp, target_xp=torch)     
>>> obs, _ = env.reset(seed=123)                                
>>> type(obs)                                                   
<class 'torch.Tensor'>
>>> action = torch.tensor(env.action_space.sample())            
>>> obs, reward, terminated, truncated, info = env.step(action) 
>>> type(obs)                                                   
<class 'torch.Tensor'>
>>> type(reward)                                                
<class 'float'>
>>> type(terminated)                                            
<class 'bool'>
>>> type(truncated)                                             
<class 'bool'>
```

Change logs:
: * v1.2.0 - Initially added

* **Parameters:**
  * **env** – The Array API compatible environment to wrap
  * **env_xp** – The Array API framework the environment is on
  * **target_xp** – The Array API framework to convert to
  * **env_device** – The device the environment is on
  * **target_device** – The device on which Arrays should be returned

### *class* gymnasium.wrappers.JaxToNumpy(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType])

Wraps a Jax-based environment such that it can be interacted with NumPy arrays.

Actions must be provided as numpy arrays and observations will be returned as numpy arrays.
A vector version of the wrapper exists, [`gymnasium.wrappers.vector.JaxToNumpy`](../vector/wrappers.md#gymnasium.wrappers.vector.JaxToNumpy).

### Notes

The Jax To Numpy and Numpy to Jax conversion does not guarantee a roundtrip (jax -> numpy -> jax) and vice versa.
The reason for this is jax does not support non-array values, therefore numpy `int_32(5) -> DeviceArray([5], dtype=jnp.int23)`

### Example

```pycon
>>> import gymnasium as gym                                     
>>> env = gym.make("JaxEnv-vx")                                 
>>> env = JaxToNumpy(env)                                       
>>> obs, _ = env.reset(seed=123)                                
>>> type(obs)                                                   
<class 'numpy.ndarray'>
>>> action = env.action_space.sample()                          
>>> obs, reward, terminated, truncated, info = env.step(action) 
>>> type(obs)                                                   
<class 'numpy.ndarray'>
>>> type(reward)                                                
<class 'float'>
>>> type(terminated)                                            
<class 'bool'>
>>> type(truncated)                                             
<class 'bool'>
```

Change logs:
: * v1.0.0 - Initially added

* **Parameters:**
  **env** – the jax environment to wrap

### *class* gymnasium.wrappers.JaxToTorch(env: [Env](../env.md#gymnasium.Env), device: str | device | None = None)

Wraps a Jax-based environment so that it can be interacted with PyTorch Tensors.

Actions must be provided as PyTorch Tensors and observations will be returned as PyTorch Tensors.
A vector version of the wrapper exists, [`gymnasium.wrappers.vector.JaxToTorch`](../vector/wrappers.md#gymnasium.wrappers.vector.JaxToTorch).

#### NOTE
For `rendered` this is returned as a NumPy array not a pytorch Tensor.

### Example

```pycon
>>> import torch                                                
>>> import gymnasium as gym                                     
>>> env = gym.make("JaxEnv-vx")                                 
>>> env = JaxtoTorch(env)                                       
>>> obs, _ = env.reset(seed=123)                                
>>> type(obs)                                                   
<class 'torch.Tensor'>
>>> action = torch.tensor(env.action_space.sample())            
>>> obs, reward, terminated, truncated, info = env.step(action) 
>>> type(obs)                                                   
<class 'torch.Tensor'>
>>> type(reward)                                                
<class 'float'>
>>> type(terminated)                                            
<class 'bool'>
>>> type(truncated)                                             
<class 'bool'>
```

Change logs:
: * v1.0.0 - Initially added

* **Parameters:**
  * **env** – The Jax-based environment to wrap
  * **device** – The device the torch Tensors should be moved to

### *class* gymnasium.wrappers.NumpyToTorch(env: [Env](../env.md#gymnasium.Env), device: str | device | None = None)

Wraps a NumPy-based environment such that it can be interacted with PyTorch Tensors.

Actions must be provided as PyTorch Tensors and observations will be returned as PyTorch Tensors.
A vector version of the wrapper exists, [`gymnasium.wrappers.vector.NumpyToTorch`](../vector/wrappers.md#gymnasium.wrappers.vector.NumpyToTorch).

#### NOTE
For `rendered` this is returned as a NumPy array not a pytorch Tensor.

### Example

```pycon
>>> import torch
>>> import gymnasium as gym
>>> env = gym.make("CartPole-v1")
>>> env = NumpyToTorch(env)
>>> obs, _ = env.reset(seed=123)
>>> type(obs)
<class 'torch.Tensor'>
>>> action = torch.tensor(env.action_space.sample())
>>> obs, reward, terminated, truncated, info = env.step(action)
>>> type(obs)
<class 'torch.Tensor'>
>>> type(reward)
<class 'float'>
>>> type(terminated)
<class 'bool'>
>>> type(truncated)
<class 'bool'>
```

Change logs:
: * v1.0.0 - Initially added

* **Parameters:**
  * **env** – The NumPy-based environment to wrap
  * **device** – The device the torch Tensors should be moved to
