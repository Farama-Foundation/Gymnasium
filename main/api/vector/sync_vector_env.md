# SyncVectorEnv

### *class* gymnasium.vector.SyncVectorEnv(env_fns: Sequence[Callable[[], [Env](../env.md#gymnasium.Env)]], copy: bool = True, observation_mode: str | tuple[[Space](../spaces.md#gymnasium.spaces.Space), [Space](../spaces.md#gymnasium.spaces.Space)] = 'same', autoreset_mode: str | [AutoresetMode](autoreset.md#gymnasium.vector.AutoresetMode) = AutoresetMode.NEXT_STEP)

Vectorized environment that serially runs multiple environments.

### Example

```pycon
>>> import gymnasium as gym
>>> envs = gym.make_vec("Pendulum-v1", num_envs=2, vectorization_mode="sync")
>>> envs
SyncVectorEnv(Pendulum-v1, num_envs=2)
>>> envs = gym.vector.SyncVectorEnv([
...     lambda: gym.make("Pendulum-v1", g=9.81),
...     lambda: gym.make("Pendulum-v1", g=1.62)
... ])
>>> envs
SyncVectorEnv(num_envs=2)
>>> obs, infos = envs.reset(seed=42)
>>> obs
array([[-0.14995256,  0.9886932 , -0.12224312],
       [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32)
>>> infos
{}
>>> _ = envs.action_space.seed(42)
>>> actions = envs.action_space.sample()
>>> obs, rewards, terminates, truncates, infos = envs.step(actions)
>>> obs
array([[-0.1878752 ,  0.98219293,  0.7695615 ],
       [ 0.6102389 ,  0.79221743, -0.8498053 ]], dtype=float32)
>>> rewards
array([-2.96562607, -0.99902063])
>>> terminates
array([False, False])
>>> truncates
array([False, False])
>>> infos
{}
>>> envs.close()
```

* **Parameters:**
  * **env_fns** – iterable of callable functions that create the environments.
  * **copy** – If `True`, then the [`reset()`](#gymnasium.vector.SyncVectorEnv.reset) and [`step()`](#gymnasium.vector.SyncVectorEnv.step) methods return a copy of the observations.
  * **observation_mode** – Defines how environment observation spaces should be batched. ‘same’ defines that there should be `n` copies of identical spaces.
    ‘different’ defines that there can be multiple observation spaces with the same length but different high/low values batched together. Passing a `Space` object
    allows the user to set some custom observation space mode not covered by ‘same’ or ‘different.’
  * **autoreset_mode** – When sub-environments are reset after their episode ends, either an [`AutoresetMode`](autoreset.md#gymnasium.vector.AutoresetMode)
    or its string value (`"NextStep"`, `"SameStep"` or `"Disabled"`), recorded in `metadata["autoreset_mode"]`.
    See [https://gymnasium.farama.org/api/vector/autoreset/](https://gymnasium.farama.org/api/vector/autoreset/) for more information.
* **Raises:**
  **RuntimeError** – If the observation space of some sub-environment does not match observation_space
      (or, by default, the observation space of the first sub-environment).

#### reset(\*, seed: int | list[int | None] | None = None, options: dict[str, Any] | None = None) → tuple[ObsType, dict[str, Any]]

Resets each of the sub-environments and concatenate the results together.

* **Parameters:**
  * **seed** – Seeds used to reset the sub-environments, either
    \* `None` - random seeds for all environment
    \* `int` - `[seed, seed+1, ..., seed+n]`
    \* List of ints - `[1, 2, 3, ..., n]`
  * **options** – Option information used for each sub-environment, `options["reset_mask"]`, a boolean array of shape `(num_envs,)`,
    resets only the masked sub-environments, as needed for `AutoresetMode.DISABLED`.
* **Returns:**
  **Concatenated observations and info from each sub-environment**

#### step(actions: ActType) → tuple[ObsType, NDArray[float64], NDArray[bool], NDArray[bool], dict[str, Any]]

Steps through each of the environments returning the batched results.

* **Returns:**
  **The batched environment step results**

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

Calls a sub-environment method with name and applies args and kwargs.

* **Parameters:**
  * **name** – The method name
  * **\*args** – The method args
  * **\*\*kwargs** – The method kwargs
* **Returns:**
  **Tuple of results**

#### get_attr(name: str) → tuple[Any, ...]

Get a property from each parallel environment.

* **Parameters:**
  **name** (*str*) – Name of the property to get from each individual environment.
* **Returns:**
  **The property with name**

#### set_attr(name: str, values: list[Any] | tuple[Any, ...] | Any) → None

Sets an attribute of the sub-environments.

* **Parameters:**
  * **name** – The property name to change
  * **values** – Values of the property to be set to. If `values` is a list or
    tuple, then it corresponds to the values for each individual
    environment, otherwise, a single value is set for all environments.
* **Raises:**
  **ValueError** – Values must be a list or tuple with length equal to the number of environments.

## Additional methods

#### *property* SyncVectorEnv.np_random *: tuple[Generator, ...]*

Returns a tuple of the numpy random number generators for the wrapped envs.

#### *property* SyncVectorEnv.np_random_seed *: tuple[int, ...]*

Returns a tuple of np random seeds for the wrapped envs.
