# Make and register

Gymnasium allows users to automatically load environments, pre-wrapped with several important wrappers through the [`gymnasium.make()`](#gymnasium.make) function. To do this, the environment must be registered prior with [`gymnasium.register()`](#gymnasium.register). To get the environment specifications for a registered environment, use [`gymnasium.spec()`](#gymnasium.spec) and to print the whole registry, use [`gymnasium.pprint_registry()`](#gymnasium.pprint_registry).
For environments registered by a third-party package on import, [`gymnasium.register_envs()`](#gymnasium.register_envs) can be used so that IDEs and linters don’t treat the import as unused.

### gymnasium.make(id: str | [EnvSpec](#gymnasium.envs.registration.EnvSpec), max_episode_steps: int | None = None, disable_env_checker: bool | None = None, \*\*kwargs: Any) → [Env](env.md#gymnasium.Env)

Creates an environment previously registered with [`gymnasium.register()`](#gymnasium.register) or a `EnvSpec`.

To find all available environments use `gymnasium.envs.registry.keys()` for all valid ids.

* **Parameters:**
  * **id** – A string for the environment id or a `EnvSpec`. Optionally if using a string, a module to import can be included, e.g. `'module:Env-v0'`.
    This is equivalent to importing the module first to register the environment followed by making the environment.
  * **max_episode_steps** – Maximum length of an episode, can override the registered `EnvSpec` `max_episode_steps`
    with the value being passed to [`gymnasium.wrappers.TimeLimit`](wrappers/misc_wrappers.md#gymnasium.wrappers.TimeLimit).
    Using `max_episode_steps=-1` will not apply the wrapper to the environment.
  * **disable_env_checker** – If to add [`gymnasium.wrappers.PassiveEnvChecker`](wrappers/misc_wrappers.md#gymnasium.wrappers.PassiveEnvChecker), `None` will default to the
    `EnvSpec` `disable_env_checker` value otherwise use this value will be used.
  * **kwargs** – Additional arguments to pass to the environment constructor.
* **Returns:**
  **An instance of the environment with wrappers applied.**
* **Raises:**
  [**Error**](utils.md#gymnasium.error.Error) – If the `id` doesn’t exist in the `registry`

Changelogs:
: v1.0.0 - autoreset and apply_api_compatibility was removed

### gymnasium.make_vec(id: str | [EnvSpec](#gymnasium.envs.registration.EnvSpec), num_envs: int = 1, vectorization_mode: [VectorizeMode](#gymnasium.VectorizeMode) | str | None = None, vector_kwargs: dict[str, Any] | None = None, wrappers: Sequence[Callable[[[Env](env.md#gymnasium.Env)], [Wrapper](wrappers.md#gymnasium.Wrapper)]] | None = None, \*\*kwargs: Any) → [VectorEnv](vector.md#gymnasium.vector.VectorEnv)

Create a vector environment according to the given ID.

To find all available environments use [`gymnasium.pprint_registry()`](#gymnasium.pprint_registry) or `gymnasium.registry.keys()` for all valid ids.
We refer to the Vector environment as the vectorizor while the environment being vectorized is the base or vectorized environment (`vectorizor(vectorized env)`).

* **Parameters:**
  * **id** – Name of the environment. Optionally, a module to import can be included, e.g. ‘module:Env-v0’
  * **num_envs** – Number of environments to create
  * **vectorization_mode** – The vectorization method used, defaults to `None` such that if env id’ spec has a `vector_entry_point` (not `None`),
    this is first used otherwise defaults to `sync` to use the [`gymnasium.vector.SyncVectorEnv`](vector/sync_vector_env.md#gymnasium.vector.SyncVectorEnv).
    Valid modes are `"async"`, `"sync"` or `"vector_entry_point"`. Recommended to use the [`VectorizeMode`](#gymnasium.VectorizeMode) enum rather than strings.
  * **vector_kwargs** – Additional arguments to pass to the vectorizor environment constructor, i.e., `SyncVectorEnv(..., **vector_kwargs)`.
  * **wrappers** – A sequence of wrapper functions to apply to the base environment. Can only be used in `"sync"` or `"async"` mode.
  * **\*\*kwargs** – Additional arguments passed to the base environment constructor.
* **Returns:**
  **An instance of the environment.**
* **Raises:**
  [**Error**](utils.md#gymnasium.error.Error) – If the `id` doesn’t exist then an error is raised

### gymnasium.register(id: str, entry_point: EnvCreator | str | None = None, reward_threshold: float | None = None, nondeterministic: bool = False, max_episode_steps: int | None = None, order_enforce: bool = True, disable_env_checker: bool = False, additional_wrappers: tuple[[WrapperSpec](#gymnasium.envs.registration.WrapperSpec), ...] = (), vector_entry_point: VectorEnvCreator | str | None = None, kwargs: dict | None = None) → None

Registers an environment in gymnasium with an `id` to use with [`gymnasium.make()`](#gymnasium.make) with the `entry_point` being a string or callable for creating the environment.

The `id` parameter corresponds to the name of the environment, with the syntax as follows:
`[namespace/](env_name)[-v(version)]` where `namespace` and `-v(version)` is optional.

It takes arbitrary keyword arguments, which are passed to the `EnvSpec` `kwargs` parameter.

* **Parameters:**
  * **id** – The environment id
  * **entry_point** – The entry point for creating the environment
  * **reward_threshold** – The reward threshold considered for an agent to have learnt the environment
  * **nondeterministic** – If the environment is nondeterministic (even with knowledge of the initial seed and all actions, the same state cannot be reached)
  * **max_episode_steps** – The maximum number of episodes steps before truncation. Used by the [`gymnasium.wrappers.TimeLimit`](wrappers/misc_wrappers.md#gymnasium.wrappers.TimeLimit) wrapper if not `None`.
  * **order_enforce** – If to enable the order enforcer wrapper to ensure users run functions in the correct order.
    If `True`, then the [`gymnasium.wrappers.OrderEnforcing`](wrappers/misc_wrappers.md#gymnasium.wrappers.OrderEnforcing) is applied to the environment.
  * **disable_env_checker** – If to disable the [`gymnasium.wrappers.PassiveEnvChecker`](wrappers/misc_wrappers.md#gymnasium.wrappers.PassiveEnvChecker) to the environment.
  * **additional_wrappers** – Additional wrappers to apply the environment.
  * **vector_entry_point** – The entry point for creating the vector environment
  * **kwargs** – arbitrary keyword arguments which are passed to the environment constructor on initialisation.

Changelogs:
: v1.0.0 - autoreset and apply_api_compatibility parameter was removed

### gymnasium.register_envs(env_module: ModuleType) → None

A no-op function such that it appears to IDEs and linters that a module is used.

Third-party packages often register their environments when imported. Passing the imported module to this
function stops the import from being flagged or removed as unused.

### Example

```pycon
>>> import ale_py  
>>> gym.register_envs(ale_py)  
>>> env = gym.make("ALE/Pong-v5")  
```

### gymnasium.spec(env_id: str) → [EnvSpec](#gymnasium.envs.registration.EnvSpec)

Retrieve the `EnvSpec` for the environment id from the `registry`.

* **Parameters:**
  **env_id** – The environment id with the expected format of `[(namespace)/]id[-v(version)]`
* **Returns:**
  **The environment spec if it exists**
* **Raises:**
  [**Error**](utils.md#gymnasium.error.Error) – If the environment id doesn’t exist

### gymnasium.pprint_registry(print_registry: dict[str, [EnvSpec](#gymnasium.envs.registration.EnvSpec)] = registry, \*, num_cols: int = 3, exclude_namespaces: list[str] | None = None, disable_print: bool = False) → str | None

Pretty prints all environments in the `registry`.

#### NOTE
All arguments are keyword only

* **Parameters:**
  * **print_registry** – Environment registry to be printed. By default, `registry`
  * **num_cols** – Number of columns to arrange environments in, for display.
  * **exclude_namespaces** – A list of namespaces to be excluded from printing. Helpful if only ALE environments are wanted.
  * **disable_print** – Whether to return a string of all the namespaces and environment IDs
    or to print the string to console.

## Core variables

### *class* gymnasium.envs.registration.EnvSpec(id: str, entry_point: ~gymnasium.envs.registration.EnvCreator | str | None = None, reward_threshold: float | None = None, nondeterministic: bool = False, max_episode_steps: int | None = None, order_enforce: bool = True, disable_env_checker: bool = False, kwargs: dict[str, ~typing.Any] = <factory>, additional_wrappers: tuple[~gymnasium.envs.registration.WrapperSpec, ...] = <factory>, vector_entry_point: ~gymnasium.envs.registration.VectorEnvCreator | str | None = None)

A specification for creating environments with [`gymnasium.make()`](#gymnasium.make).

* **id**: The string used to create the environment with [`gymnasium.make()`](#gymnasium.make)
* **entry_point**: A string for the environment location, `(import path):(environment name)` or a function that creates the environment.
* **reward_threshold**: The reward threshold for completing the environment.
* **nondeterministic**: If the observation of an environment cannot be repeated with the same initial state, random number generator state and actions.
* **max_episode_steps**: The max number of steps that the environment can take before truncation
* **order_enforce**: If to enforce the order of [`gymnasium.Env.reset()`](env.md#gymnasium.Env.reset) before [`gymnasium.Env.step()`](env.md#gymnasium.Env.step) and [`gymnasium.Env.render()`](env.md#gymnasium.Env.render) functions
* **disable_env_checker**: If to disable the environment checker wrapper in [`gymnasium.make()`](#gymnasium.make), by default False (runs the environment checker)
* **kwargs**: Additional keyword arguments passed to the environment during initialisation
* **additional_wrappers**: A tuple of additional wrappers applied to the environment (WrapperSpec)
* **vector_entry_point**: The location of the vectorized environment to create from

Changelogs:
: v1.0.0 - Autoreset attribute removed

#### make(\*\*kwargs: Any) → [Env](env.md#gymnasium.Env)

Calls `make` using the environment spec and any keyword arguments.

#### to_json() → str

Converts the environment spec into a json compatible string.

* **Returns:**
  **A jsonified string for the environment spec**

#### *static* from_json(json_env_spec: str) → [EnvSpec](#gymnasium.envs.registration.EnvSpec)

Converts a JSON string into a specification stack.

* **Parameters:**
  **json_env_spec** – A JSON string representing the env specification.
* **Returns:**
  **An environment spec**

#### pprint(disable_print: bool = False, include_entry_points: bool = False, print_all: bool = False) → str | None

Pretty prints the environment spec.

* **Parameters:**
  * **disable_print** – If to disable print and return the output
  * **include_entry_points** – If to include the entry_points in the output
  * **print_all** – If to print all information, including variables with default values
* **Returns:**
  **If \`\`disable_print is True\`\` a string otherwise \`\`None\`\`**

### *class* gymnasium.envs.registration.WrapperSpec(name: str, entry_point: str, kwargs: dict[str, Any] | None)

A specification for recording wrapper configs.

* name: The name of the wrapper.
* entry_point: The location of the wrapper to create from.
* kwargs: Additional keyword arguments passed to the wrapper. If the wrapper doesn’t inherit from EzPickle then this is `None`

### *class* gymnasium.VectorizeMode(value, names=\_not_given, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

All possible vectorization modes used in [`gymnasium.make_vec()`](#gymnasium.make_vec), passed as its `vectorization_mode` argument.

#### ASYNC *= 'async'*

#### SYNC *= 'sync'*

#### VECTOR_ENTRY_POINT *= 'vector_entry_point'*

#### gymnasium.envs.registration.registry

The Global registry for gymnasium which is where environment specifications are stored by [`gymnasium.register()`](#gymnasium.register) and from which [`gymnasium.make()`](#gymnasium.make) is used to create environments.

#### gymnasium.envs.registration.current_namespace

The current namespace when creating or registering environments. This is by default `None` but with [`namespace()`](#gymnasium.envs.registration.namespace) this can be modified to automatically set the environment id namespace.

## Additional functions

### gymnasium.envs.registration.get_env_id(ns: str | None, name: str, version: int | None) → str

Get the full env ID given a name and (optional) version and namespace. Inverse of [`parse_env_id()`](#gymnasium.envs.registration.parse_env_id).

* **Parameters:**
  * **ns** – The environment namespace
  * **name** – The environment name
  * **version** – The environment version
* **Returns:**
  **The environment id**

### gymnasium.envs.registration.parse_env_id(env_id: str) → tuple[str | None, str, int | None]

Parse environment ID string format - `[namespace/](env-name)[-v(version)]` where the namespace and version are optional.

* **Parameters:**
  **env_id** – The environment id to parse
* **Returns:**
  **A tuple of environment namespace, environment name and version number**
* **Raises:**
  [**Error**](utils.md#gymnasium.error.Error) – If the environment id is not valid environment regex

### gymnasium.envs.registration.find_highest_version(ns: str | None, name: str) → int | None

Finds the highest registered version of the environment given the namespace and name in the registry.

* **Parameters:**
  * **ns** – The environment namespace
  * **name** – The environment name (id)
* **Returns:**
  **The highest version of an environment with matching namespace and name, otherwise \`\`None\`\` is returned.**

### gymnasium.envs.registration.namespace(ns: str) → Generator[None, None, None]

Context manager for modifying the current namespace.

### gymnasium.envs.registration.load_env_creator(name: str) → EnvCreator | VectorEnvCreator

Loads an environment with name of style `"(import path):(environment name)"` and returns the environment creation function, normally the environment class type.

* **Parameters:**
  **name** – The environment name
* **Returns:**
  **The environment constructor for the given environment name.**
