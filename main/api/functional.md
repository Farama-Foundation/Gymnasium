# Functional Env

#### WARNING
The functional API is experimental, in `gymnasium.experimental`, and may change in future releases.

### *class* gymnasium.experimental.functional.FuncEnv(options: dict[str, Any] | None = None)

Base class (template) for functional envs.

This API is meant to be used in a stateless manner, with the environment state being passed around explicitly.
That being said, nothing here prevents users from using the environment statefully, it’s just not recommended.
A functional env consists of the following functions (in this case, instance methods):

> * initial: returns the initial state of the POMDP
> * observation: returns the observation in a given state
> * transition: returns the next state after taking an action in a given state
> * reward: returns the reward for a given (state, action, next_state) tuple
> * terminal: returns whether a given state is terminal
> * state_info: optional, returns a dict of info about a given state
> * step_info: optional, returns a dict of info about a given (state, action, next_state) tuple

The class-based structure serves the purpose of allowing environment constants to be defined in the class,
and then using them by name in the code itself.

For the moment, this is predominantly for internal use. This API is likely to change, but in the future
we intend to flesh it out and officially expose it to end users.

Initialize the environment constants.

#### transform(func: Callable[[Callable], Callable])

Functional transformations.

#### initial(rng: Any, params: Params | None = None) → StateType

Generates the initial state of the environment with a random number generator.

#### transition(state: StateType, action: ActType, rng: Any, params: Params | None = None) → StateType

Updates (transitions) the state with an action and random number generator.

#### observation(state: StateType, rng: Any, params: Params | None = None) → ObsType

Generates an observation for a given state of an environment.

#### reward(state: StateType, action: ActType, next_state: StateType, rng: Any, params: Params | None = None) → RewardType

Computes the reward for a given transition between state, action to next_state.

#### terminal(state: StateType, rng: Any, params: Params | None = None) → TerminalType

Returns if the state is a final terminal state.

#### state_info(state: StateType, params: Params | None = None) → dict

Info dict about a single state.

#### transition_info(state: StateType, action: ActType, next_state: StateType, params: Params | None = None) → dict

Info dict about a full transition.

#### render_init(params: Params | None = None, \*\*kwargs) → RenderStateType

Initialize the render state.

#### render_image(state: StateType, render_state: RenderStateType, params: Params | None = None) → tuple[RenderStateType, ndarray]

Show the state.

#### render_close(render_state: RenderStateType, params: Params | None = None)

Close the render state.

## Converting Jax-based functional environments to standard Env and VectorEnv

### *class* gymnasium.envs.functional_jax_env.FunctionalJaxEnv(func_env: [FuncEnv](#gymnasium.experimental.functional.FuncEnv), metadata: dict[str, Any] | None = None, render_mode: str | None = None, spec: [EnvSpec](registry.md#gymnasium.envs.registration.EnvSpec) | None = None)

A conversion layer for jax-based environments.

Initialize the environment from a FuncEnv.

#### reset(\*, seed: int | None = None, options: dict | None = None)

Resets the environment using the seed.

#### step(action: ActType)

Steps through the environment using the action.

#### render()

Returns the render state if render_mode is “rgb_array”.

### *class* gymnasium.envs.functional_jax_env.FunctionalJaxVectorEnv(func_env: [FuncEnv](#gymnasium.experimental.functional.FuncEnv)[StateType, ObsType, ActType, Any, Any, Any, Any], num_envs: int, max_episode_steps: int = 0, metadata: dict[str, Any] | None = None, render_mode: str | None = None, spec: [EnvSpec](registry.md#gymnasium.envs.registration.EnvSpec) | None = None)

A conversion layer for jax-based functional environments to a [`gymnasium.vector.VectorEnv`](vector.md#gymnasium.vector.VectorEnv).

The functional environment’s methods are vectorized with `jax.vmap()`, so all `num_envs` sub-environments
are stepped in a single call and the observations, rewards, terminations and truncations are jax arrays.

* **Parameters:**
  * **func_env** – The functional environment to vectorize
  * **num_envs** – The number of sub-environments
  * **max_episode_steps** – The number of steps after which a sub-environment is truncated, `0` for no time limit
  * **metadata** – The environment metadata, defaults to `{"autoreset_mode": AutoresetMode.NEXT_STEP}`
  * **render_mode** – The render mode, only `"rgb_array"` is supported
  * **spec** – The environment spec

#### reset(\*, seed: int | None = None, options: dict | None = None)

Resets the environment.

#### step(action: ActType)

Steps through the environment using the action.

#### render()

Returns the render state if render_mode is “rgb_array”.
