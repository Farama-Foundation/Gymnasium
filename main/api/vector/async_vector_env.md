# AsyncVectorEnv

### *class* gymnasium.vector.AsyncVectorEnv(env_fns: Sequence[Callable[[], [Env](../env.md#gymnasium.Env)]], shared_memory: bool = True, copy: bool = True, context: str | None = None, daemon: bool = True, worker: Callable[[int, Callable[[], [Env](../env.md#gymnasium.Env)], Connection, Connection, bool, Queue], None] | None = None, observation_mode: str | [Space](../spaces.md#gymnasium.spaces.Space) = 'same', autoreset_mode: str | [AutoresetMode](autoreset.md#gymnasium.vector.AutoresetMode) = AutoresetMode.NEXT_STEP, max_concurrency: int | None = None)

Vectorized environment that runs multiple environments in parallel.

It uses `multiprocessing` processes, and pipes for communication.

### Example

```pycon
>>> import gymnasium as gym
>>> envs = gym.make_vec("Pendulum-v1", num_envs=2, vectorization_mode="async")
>>> envs
AsyncVectorEnv(Pendulum-v1, num_envs=2)
>>> envs = gym.vector.AsyncVectorEnv([
...     lambda: gym.make("Pendulum-v1", g=9.81),
...     lambda: gym.make("Pendulum-v1", g=1.62)
... ])
>>> envs
AsyncVectorEnv(num_envs=2)
>>> observations, infos = envs.reset(seed=42)
>>> observations
array([[-0.14995256,  0.9886932 , -0.12224312],
       [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32)
>>> infos
{}
>>> _ = envs.action_space.seed(123)
>>> observations, rewards, terminations, truncations, infos = envs.step(envs.action_space.sample())
>>> observations
array([[-0.1851753 ,  0.98270553,  0.714599  ],
       [ 0.6193494 ,  0.7851154 , -1.0808398 ]], dtype=float32)
>>> rewards
array([-2.96495728, -1.00214607])
>>> terminations
array([False, False])
>>> truncations
array([False, False])
>>> infos
{}
```

* **Parameters:**
  * **env_fns** – Functions that create the environments.
  * **shared_memory** – If `True`, then the observations from the worker processes are communicated back through
    shared variables. This can improve the efficiency if the observations are large (e.g. images).
  * **copy** – If `True`, then the [`AsyncVectorEnv.reset()`](#gymnasium.vector.AsyncVectorEnv.reset) and [`AsyncVectorEnv.step()`](#gymnasium.vector.AsyncVectorEnv.step) methods
    return a copy of the observations.
  * **context** – Context for multiprocessing. If `None`, then the default context is used.
  * **daemon** – If `True`, then subprocesses have `daemon` flag turned on; that is, they will quit if
    the head process quits. However, `daemon=True` prevents subprocesses to spawn children,
    so for some environments you may want to have it set to `False`.
  * **worker** – If set, then use that worker in a subprocess instead of a default one.
    Can be useful to override some inner vector env logic, for instance, how resets on termination or truncation are handled. See `_async_worker` for the expected signature.
  * **observation_mode** – Defines how environment observation spaces should be batched. ‘same’ defines that there should be `n` copies of identical spaces.
    ‘different’ defines that there can be multiple observation spaces with different parameters though requires the same shape and dtype,
    warning, may raise unexpected errors. Passing a `Tuple[Space, Space]` object allows defining a custom `single_observation_space` and
    `observation_space`, warning, may raise unexpected errors.
  * **autoreset_mode** – When sub-environments are reset after their episode ends, either an [`AutoresetMode`](autoreset.md#gymnasium.vector.AutoresetMode)
    or its string value (`"NextStep"`, `"SameStep"` or `"Disabled"`), recorded in `metadata["autoreset_mode"]`.
    See [https://gymnasium.farama.org/api/vector/autoreset/](https://gymnasium.farama.org/api/vector/autoreset/) for more information.
  * **max_concurrency** – The maximum number of environments that can be executing (i.e. in `reset` or `step`)
    at any one time. If `None` (default), no limit is applied and all environments are executed in
    parallel. When set, a shared semaphore limits how many worker processes can run an environment call
    simultaneously, which can improve performance when the number of environments exceeds the number of
    CPU cores (e.g. for physics simulators such as MuJoCo). The limit applies only to `reset`
    and `step`; every other call (`render()`, [`call()`](#gymnasium.vector.AsyncVectorEnv.call), [`get_attr()`](#gymnasium.vector.AsyncVectorEnv.get_attr) and
    [`set_attr()`](#gymnasium.vector.AsyncVectorEnv.set_attr)) runs in all sub-environments simultaneously.

#### WARNING
worker is an advanced mode option. It provides a high degree of flexibility and a high chance
to shoot yourself in the foot; thus, if you are writing your own worker, it is recommended to start
from the code for `_worker` (or `_async_worker`) method, and add changes.

* **Raises:**
  * **RuntimeError** – If the observation space of some sub-environment does not match observation_space
        (or, by default, the observation space of the first sub-environment).
  * **ValueError** – If observation_space is a custom space (i.e. not a default space in Gym,
        such as gymnasium.spaces.Box, gymnasium.spaces.Discrete, or gymnasium.spaces.Dict) and shared_memory is True.
  * **ValueError** – If `max_concurrency` is not a positive integer, or if a custom `worker` does not
        accept the `semaphore` keyword argument.

#### reset(\*, seed: int | list[int | None] | None = None, options: dict[str, Any] | None = None) → tuple[ObsType, dict[str, Any]]

Resets all sub-environments in parallel and return a batch of concatenated observations and info.

* **Parameters:**
  * **seed** – The environment reset seeds
  * **options** – Option information used for each sub-environment, `options["reset_mask"]`, a boolean array of shape `(num_envs,)`,
    resets only the masked sub-environments, as needed for `AutoresetMode.DISABLED`.
* **Returns:**
  **A batch of observations and info from the vectorized environment.**

#### step(actions: ActType) → tuple[ObsType, NDArray[float64], NDArray[bool], NDArray[bool], dict[str, Any]]

Take an action for each parallel environment.

* **Parameters:**
  **actions** – element of `action_space` batch of actions.
* **Returns:**
  **Batch of** (*observations, rewards, terminations, truncations, infos*)

#### close(\*\*kwargs: Any) → None

Close all parallel environments and release resources.

It also closes all the existing image viewers, then calls `close_extras()` and set
`closed` as `True`.

#### WARNING
This function itself does not close the environments, it should be handled
in `close_extras()`. This is generic for both synchronous and asynchronous
vectorized environments.

#### NOTE
This will be automatically called when garbage collected or program exited.

* **Parameters:**
  **\*\*kwargs** – Keyword arguments passed to `close_extras()`

#### call(name: str, \*args: Any, \*\*kwargs: Any) → tuple[Any, ...]

Call a method from each parallel environment with args and kwargs.

* **Parameters:**
  * **name** (*str*) – Name of the method or property to call.
  * **\*args** – Position arguments to apply to the method call.
  * **\*\*kwargs** – Keyword arguments to apply to the method call.
* **Returns:**
  **List of the results of the individual calls to the method or property for each environment.**

#### get_attr(name: str) → tuple[Any, ...]

Get a property from each parallel environment.

* **Parameters:**
  **name** (*str*) – Name of the property to be get from each individual environment.
* **Returns:**
  **The property with name**

#### set_attr(name: str, values: list[Any] | tuple[Any] | object) → None

Sets an attribute of the sub-environments.

* **Parameters:**
  * **name** – Name of the property to be set in each individual environment.
  * **values** – Values of the property to be set to. If `values` is a list or
    tuple, then it corresponds to the values for each individual
    environment, otherwise a single value is set for all environments.
* **Raises:**
  * **ValueError** – Values must be a list or tuple with length equal to the number of environments.
  * [**AlreadyPendingCallError**](../utils.md#gymnasium.error.AlreadyPendingCallError) – Calling [`set_attr()`](#gymnasium.vector.AsyncVectorEnv.set_attr) while waiting for a pending call to complete.

## Asynchronous methods

[`reset()`](#gymnasium.vector.AsyncVectorEnv.reset), [`step()`](#gymnasium.vector.AsyncVectorEnv.step) and [`call()`](#gymnasium.vector.AsyncVectorEnv.call) are each implemented as a pair of methods: an `_async` method that sends the command to the worker processes and returns immediately, and a `_wait` method that blocks until the results are returned. These can be used directly to run other code while the sub-environments step.

#### AsyncVectorEnv.reset_async(seed: int | list[int | None] | None = None, options: dict[str, Any] | None = None) → None

Send calls to the [`reset`](#gymnasium.vector.AsyncVectorEnv.reset) methods of the sub-environments.

To get the results of these calls, you may invoke [`reset_wait()`](#gymnasium.vector.AsyncVectorEnv.reset_wait).

* **Parameters:**
  * **seed** – List of seeds for each environment
  * **options** – Option information used for each sub-environment, `options["reset_mask"]`, a boolean array of shape `(num_envs,)`,
    resets only the masked sub-environments, as needed for `AutoresetMode.DISABLED`.
* **Raises:**
  * [**ClosedEnvironmentError**](../utils.md#gymnasium.error.ClosedEnvironmentError) – If the environment was closed (if [`close()`](#gymnasium.vector.AsyncVectorEnv.close) was previously called).
  * [**AlreadyPendingCallError**](../utils.md#gymnasium.error.AlreadyPendingCallError) – If the environment is already waiting for a pending call to another
        method (e.g. [`step_async()`](#gymnasium.vector.AsyncVectorEnv.step_async)). This can be caused by two consecutive
        calls to [`reset_async()`](#gymnasium.vector.AsyncVectorEnv.reset_async), with no call to [`reset_wait()`](#gymnasium.vector.AsyncVectorEnv.reset_wait) in between.

#### AsyncVectorEnv.reset_wait(timeout: float | None = None) → tuple[ObsType, dict[str, Any]]

Waits for the calls triggered by [`reset_async()`](#gymnasium.vector.AsyncVectorEnv.reset_async) to finish and returns the results.

* **Parameters:**
  **timeout** – Number of seconds before the call to `reset_wait` times out. If None, the call to `reset_wait` never times out.
* **Returns:**
  **A tuple of batched observations and list of dictionaries**
* **Raises:**
  * [**ClosedEnvironmentError**](../utils.md#gymnasium.error.ClosedEnvironmentError) – If the environment was closed (if [`close()`](#gymnasium.vector.AsyncVectorEnv.close) was previously called).
  * [**NoAsyncCallError**](../utils.md#gymnasium.error.NoAsyncCallError) – If [`reset_wait()`](#gymnasium.vector.AsyncVectorEnv.reset_wait) was called without any prior call to [`reset_async()`](#gymnasium.vector.AsyncVectorEnv.reset_async).
  * **TimeoutError** – If [`reset_wait()`](#gymnasium.vector.AsyncVectorEnv.reset_wait) timed out.

#### AsyncVectorEnv.step_async(actions: ActType) → None

Send the calls to `Env.step()` to each sub-environment.

* **Parameters:**
  **actions** – Batch of actions. element of [`VectorEnv.action_space`](../vector.md#gymnasium.vector.VectorEnv.action_space)
* **Raises:**
  * [**ClosedEnvironmentError**](../utils.md#gymnasium.error.ClosedEnvironmentError) – If the environment was closed (if [`close()`](#gymnasium.vector.AsyncVectorEnv.close) was previously called).
  * [**AlreadyPendingCallError**](../utils.md#gymnasium.error.AlreadyPendingCallError) – If the environment is already waiting for a pending call to another
        method (e.g. [`reset_async()`](#gymnasium.vector.AsyncVectorEnv.reset_async)). This can be caused by two consecutive
        calls to [`step_async()`](#gymnasium.vector.AsyncVectorEnv.step_async), with no call to [`step_wait()`](#gymnasium.vector.AsyncVectorEnv.step_wait) in
        between.

#### AsyncVectorEnv.step_wait(timeout: float | None = None) → tuple[ObsType, NDArray[float64], NDArray[bool], NDArray[bool], dict[str, Any]]

Wait for the calls to [`step`](#gymnasium.vector.AsyncVectorEnv.step) in each sub-environment to finish.

* **Parameters:**
  **timeout** – Number of seconds before the call to [`step_wait()`](#gymnasium.vector.AsyncVectorEnv.step_wait) times out. If `None`, the call to [`step_wait()`](#gymnasium.vector.AsyncVectorEnv.step_wait) never times out.
* **Returns:**
  **The batched environment step information,** (*obs, reward, terminated, truncated, info*)
* **Raises:**
  * [**ClosedEnvironmentError**](../utils.md#gymnasium.error.ClosedEnvironmentError) – If the environment was closed (if [`close()`](#gymnasium.vector.AsyncVectorEnv.close) was previously called).
  * [**NoAsyncCallError**](../utils.md#gymnasium.error.NoAsyncCallError) – If [`step_wait()`](#gymnasium.vector.AsyncVectorEnv.step_wait) was called without any prior call to [`step_async()`](#gymnasium.vector.AsyncVectorEnv.step_async).
  * **TimeoutError** – If [`step_wait()`](#gymnasium.vector.AsyncVectorEnv.step_wait) timed out.

#### AsyncVectorEnv.call_async(name: str, /, \*args: object, \*\*kwargs: object) → None

Calls the method with name asynchronously and apply args and kwargs to the method.

* **Parameters:**
  * **name** – Name of the method or property to call.
  * **\*args** – Arguments to apply to the method call.
  * **\*\*kwargs** – Keyword arguments to apply to the method call.
* **Raises:**
  * [**ClosedEnvironmentError**](../utils.md#gymnasium.error.ClosedEnvironmentError) – If the environment was closed (if [`close()`](#gymnasium.vector.AsyncVectorEnv.close) was previously called).
  * [**AlreadyPendingCallError**](../utils.md#gymnasium.error.AlreadyPendingCallError) – Calling call_async while waiting for a pending call to complete

#### AsyncVectorEnv.call_wait(timeout: float | None = None) → tuple[Any, ...]

Calls all parent pipes and waits for the results.

* **Parameters:**
  **timeout** – Number of seconds before the call to [`step_wait()`](#gymnasium.vector.AsyncVectorEnv.step_wait) times out.
  If `None` (default), the call to [`step_wait()`](#gymnasium.vector.AsyncVectorEnv.step_wait) never times out.
* **Returns:**
  **List of the results of the individual calls to the method or property for each environment.**
* **Raises:**
  * [**NoAsyncCallError**](../utils.md#gymnasium.error.NoAsyncCallError) – Calling [`call_wait()`](#gymnasium.vector.AsyncVectorEnv.call_wait) without any prior call to [`call_async()`](#gymnasium.vector.AsyncVectorEnv.call_async).
  * **TimeoutError** – The call to [`call_wait()`](#gymnasium.vector.AsyncVectorEnv.call_wait) has timed out after timeout second(s).

## Additional methods

#### *property* AsyncVectorEnv.np_random *: tuple[Generator, ...]*

Returns the tuple of the numpy random number generators for the wrapped envs.

#### *property* AsyncVectorEnv.np_random_seed *: tuple[int, ...]*

Returns a tuple of np_random seeds for all the wrapped envs.
