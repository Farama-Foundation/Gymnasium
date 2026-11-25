# Utility functions

## Seeding

### gymnasium.utils.seeding.np_random(seed: int | None = None) → tuple[Generator, int]

Returns a NumPy random number generator (RNG) along with seed value from the inputted seed.

If `seed` is `None` then a **random** seed will be generated as the RNG’s initial seed.
This randomly selected seed is returned as the second value of the tuple.

This function is called in [`reset()`](env.md#gymnasium.Env.reset) to reset an environment’s initial RNG.

* **Parameters:**
  **seed** – The seed used to create the generator
* **Returns:**
  **A NumPy-based Random Number Generator and generator seed**
* **Raises:**
  [**Error**](#gymnasium.error.Error) – Seed must be a non-negative integer

## Environment checking

### gymnasium.utils.env_checker.check_env(env: [Env](env.md#gymnasium.Env), warn: bool | None = None, skip_render_check: bool = False, skip_close_check: bool = False) → None

Check that an environment follows Gymnasium’s API.

To ensure that an environment is implemented “correctly”, `check_env` checks that the [`observation_space`](env.md#gymnasium.Env.observation_space) and [`action_space`](env.md#gymnasium.Env.action_space) are correct.
Furthermore, the function will call the [`reset()`](env.md#gymnasium.Env.reset), [`step()`](env.md#gymnasium.Env.step) and [`render()`](env.md#gymnasium.Env.render) functions with a variety of values.

We highly recommend users call this function after an environment is constructed and within a project’s continuous integration to keep an environment update with Gymnasium’s API.

* **Parameters:**
  * **env** – The Gym environment that will be checked
  * **warn** – Ignored, previously silenced particular warnings
  * **skip_render_check** – Whether to skip the checks for the render method. False by default (useful for the CI)
  * **skip_close_check** – Whether to skip the checks for the close method. False by default

### gymnasium.utils.env_match.check_environments_match(env_a: [Env](env.md#gymnasium.Env), env_b: [Env](env.md#gymnasium.Env), num_steps: int, seed: int = 0, skip_obs: bool = False, skip_rew: bool = False, skip_terminal: bool = False, skip_truncated: bool = False, skip_render: bool = False, info_comparison: str = 'equivalence') → None

Checks if the environments env_a & env_b are identical.

* **Parameters:**
  * **env_a** – First environment to check.
  * **env_b** – Second environment to check.
  * **num_steps** – number of timesteps to test for, setting to 0 tests only resetting.
  * **seed** – The seed used to reset both environments and to seed env_a’s action space for sampling actions.
  * **skip_obs** – If True it does not check for equivalence of the observation space or the observations.
  * **skip_rew** – If True it does not check for equivalence of the rewards.
  * **skip_terminal** – If True it does not check for equivalence of the terminated signals.
  * **skip_truncated** – If True it does not check for equivalence of the truncated signals.
  * **skip_render** – If True it does not check for equivalent renders. Note: the render checks are automatically skipped if either environment’s render_mode is None or “human”.
  * **info_comparison** – The options are
    If “equivalence” then checks if the info\`s are identical,
    If “superset” checks if \`info_b is a (non-strict) superset of info_a
    If “keys-equivalence” checks if the info\`s keys are identical (while ignoring the values).
    If “keys-superset” checks if the \`info_b\`s keys are a superset of \`info_a’s keys.
    If “skip” no checks are made at the info.

## Visualization

### gymnasium.utils.play.play(env: [Env](env.md#gymnasium.Env)[Any, ActType], transpose: bool | None = True, fps: int | None = None, zoom: float | None = None, callback: Callable | None = None, keys_to_action: dict[\_ActionKey, ActType] | None = None, seed: int | None = None, noop: ActType | int = 0, wait_on_player: bool = False) → None

Allows the user to play the environment using a keyboard.

If playing in a turn-based environment, set wait_on_player to True.

* **Parameters:**
  * **env** – Environment to use for playing.
  * **transpose** – If this is `True`, the output of observation is transposed. Defaults to `True`.
  * **fps** – Maximum number of steps of the environment executed every second. If `None` (the default),
    `env.metadata["render_fps""]` (or 30, if the environment does not specify “render_fps”) is used.
  * **zoom** – Zoom the observation in, `zoom` amount, should be positive float
  * **callback** – 

    If a callback is provided, it will be executed after every step. It takes the following input:
    * obs_t: observation before performing action
    * obs_tp1: observation after performing action
    * action: action that was executed
    * rew: reward that was received
    * terminated: whether the environment is terminated or not
    * truncated: whether the environment is truncated or not
    * info: debug info
  * **keys_to_action** – 

    Mapping from keys pressed to action performed.
    Different formats are supported: Key combinations can either be expressed as a tuple of unicode code
    points of the keys, as a tuple of characters, or as a string where each character of the string represents
    one key.
    For example if pressing ‘w’ and space at the same time is supposed
    to trigger action number 2 then `key_to_action` dict could look like this:
    ```pycon
    >>> key_to_action = {
    ...    # ...
    ...    (ord('w'), ord(' ')): 2
    ...    # ...
    ... }
    ```

    or like this:
    ```pycon
    >>> key_to_action = {
    ...    # ...
    ...    ("w", " "): 2
    ...    # ...
    ... }
    ```

    or like this:
    ```pycon
    >>> key_to_action = {
    ...    # ...
    ...    "w ": 2
    ...    # ...
    ... }
    ```

    If `None`, default `key_to_action` mapping for that environment is used, if provided.
  * **seed** – Random seed used when resetting the environment. If None, no seed is used.
  * **noop** – The action used when no key input has been entered, or the entered key combination is unknown.
  * **wait_on_player** – Play should wait for a user action

### Example

```pycon
>>> import gymnasium as gym
>>> import numpy as np
>>> from gymnasium.utils.play import play
>>> play(gym.make("CarRacing-v3", render_mode="rgb_array"),  
...     keys_to_action={
...         "w": np.array([0, 0.7, 0], dtype=np.float32),
...         "a": np.array([-1, 0, 0], dtype=np.float32),
...         "s": np.array([0, 0, 1], dtype=np.float32),
...         "d": np.array([1, 0, 0], dtype=np.float32),
...         "wa": np.array([-1, 0.7, 0], dtype=np.float32),
...         "dw": np.array([1, 0.7, 0], dtype=np.float32),
...         "ds": np.array([1, 0, 1], dtype=np.float32),
...         "as": np.array([-1, 0, 1], dtype=np.float32),
...     },
...     noop=np.array([0, 0, 0], dtype=np.float32)
... )
```

Above code works also if the environment is wrapped, so it’s particularly useful in
verifying that the frame-level preprocessing does not render the game
unplayable.

If you wish to plot real time statistics as you play, you can use
[`PlayPlot`](#gymnasium.utils.play.PlayPlot). Here’s a sample code for plotting the reward
for last 150 steps.

```pycon
>>> from gymnasium.utils.play import PlayPlot, play
>>> def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
...        return [rew,]
>>> plotter = PlayPlot(callback, 150, ["reward"])             
>>> play(gym.make("CartPole-v1"), callback=plotter.callback)  
```

### *class* gymnasium.utils.play.PlayPlot(callback: Callable[[\_ObsT_contra, \_ObsT_contra, \_ActT_contra, float, bool, bool, dict], Iterable[Any]], horizon_timesteps: int, plot_names: list[str])

Provides a callback to create live plots of arbitrary metrics when using [`play()`](#gymnasium.utils.play.play).

This class is instantiated with a function that accepts information about a single environment transition:
: - obs_t: observation before performing action
  - obs_tp1: observation after performing action
  - action: action that was executed
  - rew: reward that was received
  - terminated: whether the environment is terminated or not
  - truncated: whether the environment is truncated or not
  - info: debug info

It should return a list of metrics that are computed from this data.
For instance, the function may look like this:

```default
>>> def compute_metrics(obs_t, obs_tp, action, reward, terminated, truncated, info):
...     return [reward, info["cumulative_reward"], np.linalg.norm(action)]
```

[`PlayPlot`](#gymnasium.utils.play.PlayPlot) provides the method [`callback()`](#gymnasium.utils.play.PlayPlot.callback) which will pass its arguments along to that function
and uses the returned values to update live plots of the metrics.

Typically, this [`callback()`](#gymnasium.utils.play.PlayPlot.callback) will be used in conjunction with [`play()`](#gymnasium.utils.play.play) to see how the metrics evolve as you play:

```default
>>> plotter = PlayPlot(compute_metrics, horizon_timesteps=200,                               
...                    plot_names=["Immediate Rew.", "Cumulative Rew.", "Action Magnitude"])
>>> play(your_env, callback=plotter.callback)                                                
```

* **Parameters:**
  * **callback** – Function that computes metrics from environment transitions
  * **horizon_timesteps** – The time horizon used for the live plots
  * **plot_names** – List of plot titles
* **Raises:**
  [**DependencyNotInstalled**](#gymnasium.error.DependencyNotInstalled) – If matplotlib is not installed

#### callback(obs_t: \_ObsT_contra, obs_tp1: \_ObsT_contra, action: \_ActT_contra, rew: float, terminated: bool, truncated: bool, info: dict) → None

The callback that calls the provided data callback and adds the data to the plots.

* **Parameters:**
  * **obs_t** – The observation at time step t
  * **obs_tp1** – The observation at time step t+1
  * **action** – The action
  * **rew** – The reward
  * **terminated** – If the environment is terminated
  * **truncated** – If the environment is truncated
  * **info** – The information from the environment

### *class* gymnasium.utils.play.PlayableGame(env: [Env](env.md#gymnasium.Env), keys_to_action: dict[tuple[int, ...], int] | None = None, zoom: float | None = None)

Wraps an environment allowing keyboard inputs to interact with the environment.

* **Parameters:**
  * **env** – The environment to play
  * **keys_to_action** – The dictionary of keyboard tuples and action value
  * **zoom** – If to zoom in on the environment render

#### process_event(event: Event) → None

Processes a PyGame event.

In particular, this function is used to keep track of which buttons are currently pressed
and to exit the [`play()`](#gymnasium.utils.play.play) function when the PyGame window is closed.

* **Parameters:**
  **event** – The event to process

## Environment pickling

### *class* gymnasium.utils.ezpickle.EzPickle(\*args: object, \*\*kwargs: object)

Objects that are pickled and unpickled via their constructor arguments.

### Example

```pycon
>>> class Animal: pass
>>> class Dog(Animal, EzPickle):
...    def __init__(self, furcolor, tailkind="bushy"):
...        Animal.__init__(self)
...        EzPickle.__init__(self, furcolor, tailkind)
```

When this object is unpickled, a new `Dog` will be constructed by passing the provided furcolor and tailkind into the constructor.
However, philosophers are still not sure whether it is still the same dog.

This is generally needed only for environments which wrap C/C++ code, such as MuJoCo and Atari.

Uses the `args` and `kwargs` from the object’s constructor for pickling.

### *class* gymnasium.utils.RecordConstructorArgs(\*, \_disable_deepcopy: bool = False, \*\*kwargs: Any)

Records all arguments passed to constructor to \_saved_kwargs.

This can be used to save and reproduce class constructor arguments.

#### NOTE
If two class inherit from RecordConstructorArgs then the first class to call RecordConstructorArgs._\_init_\_(self, …) will have
their kwargs saved will all subsequent RecordConstructorArgs._\_init_\_ being ignored.

Therefore, always call RecordConstructorArgs._\_init_\_ before the Class._\_init_\_

* **Parameters:**
  * **\_disable_deepcopy** – If to not deepcopy the kwargs passed
  * **\*\*kwargs** – Arguments to save

## Save rendering videos

### gymnasium.utils.save_video.save_video(frames: list, video_folder: str, episode_trigger: Callable[[int], bool] | None = None, step_trigger: Callable[[int], bool] | None = None, video_length: int | None = None, name_prefix: str = 'rl-video', episode_index: int = 0, step_starting_index: int = 0, save_logger: str | None = None, \*\*kwargs: Any) → None

Save videos from rendering frames.

This function extract video from a list of render frame episodes.

* **Parameters:**
  * **frames** (*List* *[**RenderFrame* *]*) – A list of frames to compose the video.
  * **video_folder** (*str*) – The folder where the recordings will be stored
  * **episode_trigger** – Function that accepts an integer and returns `True` iff a recording should be started at this episode
  * **step_trigger** – Function that accepts an integer and returns `True` iff a recording should be started at this step
  * **video_length** (*int*) – The length of recorded episodes. If it isn’t specified, the entire episode is recorded.
    Otherwise, snippets of the specified length are captured.
  * **name_prefix** (*str*) – Will be prepended to the filename of the recordings.
  * **episode_index** (*int*) – The index of the current episode.
  * **step_starting_index** (*int*) – The step index of the first frame.
  * **save_logger** – If to log the video saving progress, helpful for long videos that take a while, use “bar” to enable.
  * **\*\*kwargs** – The kwargs that will be passed to moviepy’s ImageSequenceClip.
    You need to specify either fps or duration.

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.utils.save_video import save_video
>>> env = gym.make("FrozenLake-v1", render_mode="rgb_array_list")
>>> _ = env.reset()
>>> step_starting_index = 0
>>> episode_index = 0
>>> for step_index in range(199): 
...    action = env.action_space.sample()
...    _, _, terminated, truncated, _ = env.step(action)
...
...    if terminated or truncated:
...       save_video(
...          frames=env.render(),
...          video_folder="videos",
...          fps=env.metadata["render_fps"],
...          step_starting_index=step_starting_index,
...          episode_index=episode_index
...       )
...       step_starting_index = step_index + 1
...       episode_index += 1
...       env.reset()
>>> env.close()
```

### gymnasium.utils.save_video.capped_cubic_video_schedule(episode_id: int) → bool

The default episode trigger.

This function will trigger recordings at the episode indices $\{0, 1, 4, 8, 27, ..., k^3, ..., 729, 1000, 2000, 3000, ...\}$

* **Parameters:**
  **episode_id** – The episode number
* **Returns:**
  **If to apply a video schedule number**

## Old to new step API compatibility

### gymnasium.utils.step_api_compatibility.step_api_compatibility(step_returns: tuple[ObsType | ndarray, SupportsFloat | ndarray, bool | ndarray, bool | ndarray, dict | list] | tuple[ObsType | ndarray, SupportsFloat | ndarray, bool | ndarray, dict | list], output_truncation_bool: bool = True, is_vector_env: bool = False) → tuple[ObsType | ndarray, SupportsFloat | ndarray, bool | ndarray, bool | ndarray, dict | list] | tuple[ObsType | ndarray, SupportsFloat | ndarray, bool | ndarray, dict | list]

Function to transform step returns to the API specified by `output_truncation_bool`.

Done (old) step API refers to [`step()`](env.md#gymnasium.Env.step) method returning `(observation, reward, done, info)`
Terminated Truncated (new) step API refers to [`step()`](env.md#gymnasium.Env.step) method returning `(observation, reward, terminated, truncated, info)`
(Refer to docs for details on the API change)

* **Parameters:**
  * **step_returns** (*tuple*) – Items returned by [`step()`](env.md#gymnasium.Env.step). Can be `(obs, rew, done, info)` or `(obs, rew, terminated, truncated, info)`
  * **output_truncation_bool** (*bool*) – Whether the output should return two booleans (new API) or one (old) (`True` by default)
  * **is_vector_env** (*bool*) – Whether the `step_returns` are from a vector environment
* **Returns:**
  **step_returns** (*tuple*) – Depending on `output_truncation_bool`, it can return `(obs, rew, done, info)` or `(obs, rew, terminated, truncated, info)`

### Example

This function can be used to ensure compatibility in step interfaces with conflicting API. E.g. if env is written in old API,
wrapper is written in new API, and the final step output is desired to be in old API.

```pycon
>>> import gymnasium as gym
>>> env = gym.make("CartPole-v0")
>>> _, _ = env.reset()
>>> obs, reward, done, info = step_api_compatibility(env.step(0), output_truncation_bool=False)
>>> obs, reward, terminated, truncated, info = step_api_compatibility(env.step(0), output_truncation_bool=True)
```

```pycon
>>> vec_env = gym.make_vec("CartPole-v0", vectorization_mode="sync")
>>> _, _ = vec_env.reset()
>>> obs, rewards, dones, infos = step_api_compatibility(vec_env.step([0]), is_vector_env=True, output_truncation_bool=False)
>>> obs, rewards, terminations, truncations, infos = step_api_compatibility(vec_env.step([0]), is_vector_env=True, output_truncation_bool=True)
```

### gymnasium.utils.step_api_compatibility.convert_to_terminated_truncated_step_api(step_returns: tuple[ObsType | ndarray, SupportsFloat | ndarray, bool | ndarray, dict | list] | tuple[ObsType | ndarray, SupportsFloat | ndarray, bool | ndarray, bool | ndarray, dict | list], is_vector_env: bool = False) → tuple[ObsType | ndarray, SupportsFloat | ndarray, bool | ndarray, bool | ndarray, dict | list]

Function to transform step returns to new step API irrespective of input API.

* **Parameters:**
  * **step_returns** (*tuple*) – Items returned by [`step()`](env.md#gymnasium.Env.step). Can be `(obs, rew, done, info)` or `(obs, rew, terminated, truncated, info)`
  * **is_vector_env** (*bool*) – Whether the `step_returns` are from a vector environment

### gymnasium.utils.step_api_compatibility.convert_to_done_step_api(step_returns: tuple[ObsType | ndarray, SupportsFloat | ndarray, bool | ndarray, bool | ndarray, dict | list] | tuple[ObsType | ndarray, SupportsFloat | ndarray, bool | ndarray, dict | list], is_vector_env: bool = False) → tuple[ObsType | ndarray, SupportsFloat | ndarray, bool | ndarray, dict | list]

Function to transform step returns to old step API irrespective of input API.

* **Parameters:**
  * **step_returns** (*tuple*) – Items returned by [`step()`](env.md#gymnasium.Env.step). Can be `(obs, rew, done, info)` or `(obs, rew, terminated, truncated, info)`
  * **is_vector_env** (*bool*) – Whether the `step_returns` are from a vector environment

## Runtime performance benchmark

Sometimes it is necessary to measure your environment’s runtime performance, and ensure no performance regressions take place.
These tests require manual inspection of its outputs:

### gymnasium.utils.performance.benchmark_step(env: [Env](env.md#gymnasium.Env), target_duration: int = 5, seed: int | None = None) → float

A benchmark to measure the runtime performance of step for an environment.

example usage:
: ``py
  env_old = ...
  old_throughput = benchmark_step(env_old)
  env_new = ...
  new_throughput = benchmark_step(env_old)
  slowdown = old_throughput / new_throughput
  ``

* **Parameters:**
  * **env** – the environment to benchmarked.
  * **target_duration** – the duration of the benchmark in seconds (note: it will go slightly over it).
  * **seed** – seeds the environment and action sampled.

Returns: the average steps per second.

### gymnasium.utils.performance.benchmark_vector_step(env: [VectorEnv](vector.md#gymnasium.vector.VectorEnv), target_duration: int = 5, seed: int | None = None) → float

Measure the step throughput of a vector environment.

* **Parameters:**
  * **env** – The vector environment to benchmark.
  * **target_duration** – The benchmark duration in seconds. The benchmark can run
    slightly longer while its final step completes.
  * **seed** – Seed for the environment and action space.
* **Returns:**
  **The number of individual environment steps per second.**

### gymnasium.utils.performance.benchmark_init(env_lambda: Callable[[], [Env](env.md#gymnasium.Env)], target_duration: int = 5, seed: int | None = None) → float

A benchmark to measure the initialization time and first reset.

* **Parameters:**
  * **env_lambda** – the function to initialize the environment.
  * **target_duration** – the duration of the benchmark in seconds (note: it will go slightly over it).
  * **seed** – seeds the first reset of the environment.

### gymnasium.utils.performance.benchmark_render(env: [Env](env.md#gymnasium.Env), target_duration: int = 5) → float

A benchmark to measure the time of render().

Note: does not work with render_mode=’human’
:param env: the environment to benchmarked (Note: must be renderable).
:param target_duration: the duration of the benchmark in seconds (note: it will go slightly over it).

## Errors

Set of Error classes for gymnasium.

### *exception* gymnasium.error.AlreadyPendingCallError(message: str, name: str)

Bases: `Exception`

Raised when reset, or step is called asynchronously (e.g. with reset_async, or step_async respectively), and reset_async, or step_async (respectively) is called again (without a complete call to reset_wait, or step_wait respectively).

Initialises the exception with name attributes.

### *exception* gymnasium.error.ClosedEnvironmentError

Bases: `Exception`

Trying to call reset, or step, while the environment is closed.

### *exception* gymnasium.error.CustomSpaceError

Bases: `Exception`

The space is a custom gymnasium.Space instance, and is not supported by AsyncVectorEnv with shared_memory=True.

### *exception* gymnasium.error.DependencyNotInstalled

Bases: [`Error`](#gymnasium.error.Error)

Raised when the user has not installed a dependency.

### *exception* gymnasium.error.DeprecatedEnv

Bases: [`Error`](#gymnasium.error.Error)

Raised when the user requests an env from the registry with an older version number than the latest env with the same name.

### *exception* gymnasium.error.DeprecatedWrapper

Bases: `ImportError`

Error message for importing an old version of a wrapper.

### *exception* gymnasium.error.Error

Bases: `Exception`

Error superclass.

### *exception* gymnasium.error.InvalidAction

Bases: [`Error`](#gymnasium.error.Error)

Raised when the user performs an action not contained within the action space.

### *exception* gymnasium.error.InvalidBound

Bases: [`Error`](#gymnasium.error.Error)

Raised when the clipping an array with invalid upper and/or lower bound.

### *exception* gymnasium.error.InvalidMetadata

Bases: [`Error`](#gymnasium.error.Error)

Raised when the metadata of an environment is not valid.

### *exception* gymnasium.error.InvalidProbability

Bases: [`Error`](#gymnasium.error.Error)

Raised when given an invalid value for a probability.

### *exception* gymnasium.error.MissingArgument

Bases: [`Error`](#gymnasium.error.Error)

Raised when a required argument in the initializer is missing.

### *exception* gymnasium.error.NameNotFound

Bases: [`UnregisteredEnv`](#gymnasium.error.UnregisteredEnv)

Raised when the user requests an env from the registry where the name doesn’t exist.

### *exception* gymnasium.error.NamespaceNotFound

Bases: [`UnregisteredEnv`](#gymnasium.error.UnregisteredEnv)

Raised when the user requests an env from the registry where the namespace doesn’t exist.

### *exception* gymnasium.error.NoAsyncCallError(message: str, name: str)

Bases: `Exception`

Raised when an asynchronous reset, or step is not running, but reset_wait, or step_wait (respectively) is called.

Initialises the exception with name attributes.

### *exception* gymnasium.error.RegistrationError

Bases: [`Error`](#gymnasium.error.Error)

Raised when the user attempts to register an invalid env. For example, an unversioned env when a versioned env exists.

### *exception* gymnasium.error.ResetNeeded

Bases: [`Error`](#gymnasium.error.Error)

When the order enforcing is violated, i.e. step or render is called before reset.

### *exception* gymnasium.error.UnregisteredEnv

Bases: [`Error`](#gymnasium.error.Error)

Raised when the user requests an env from the registry that does not actually exist.

### *exception* gymnasium.error.UnsupportedMode

Bases: [`Error`](#gymnasium.error.Error)

Raised when the user requests a rendering mode not supported by the environment.

### *exception* gymnasium.error.VersionNotFound

Bases: [`UnregisteredEnv`](#gymnasium.error.UnregisteredEnv)

Raised when the user requests an env from the registry where the version doesn’t exist.
