# Wrappers

<a id="module-gymnasium.wrappers"></a>

Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly.

Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular.
Importantly wrappers can be chained to combine their effects and most environments that are generated via
[`gymnasium.make()`](registry.md#gymnasium.make) will already be wrapped by default.

In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along
with (possibly optional) parameters to the wrapper’s constructor.

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import RescaleAction
>>> base_env = gym.make("Hopper-v4")
>>> base_env.action_space
Box(-1.0, 1.0, (3,), float32)
>>> wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
>>> wrapped_env.action_space
Box(0.0, 1.0, (3,), float32)
```

You can access the environment underneath the **first** wrapper by using the [`gymnasium.Wrapper.env`](#gymnasium.Wrapper.env) attribute.
As the [`gymnasium.Wrapper`](#gymnasium.Wrapper) class inherits from [`gymnasium.Env`](env.md#gymnasium.Env) then [`gymnasium.Wrapper.env`](#gymnasium.Wrapper.env) can be another wrapper.

```pycon
>>> wrapped_env
<RescaleAction<TimeLimit<OrderEnforcing<PassiveEnvChecker<HopperEnv<Hopper-v4>>>>>>
>>> wrapped_env.env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<HopperEnv<Hopper-v4>>>>>
```

The wrappers that [`gymnasium.make()`](registry.md#gymnasium.make) applies by default are, from innermost to outermost:

- [`gymnasium.wrappers.PassiveEnvChecker`](wrappers/misc_wrappers.md#gymnasium.wrappers.PassiveEnvChecker), unless `disable_env_checker=True` is passed or set in the environment spec
- [`gymnasium.wrappers.OrderEnforcing`](wrappers/misc_wrappers.md#gymnasium.wrappers.OrderEnforcing), if the environment spec’s `order_enforce` is `True` (the default)
- [`gymnasium.wrappers.TimeLimit`](wrappers/misc_wrappers.md#gymnasium.wrappers.TimeLimit), if `max_episode_steps` is passed or set in the environment spec, `max_episode_steps=-1` disables it
- the wrappers in the environment spec’s `additional_wrappers`
- [`gymnasium.wrappers.HumanRendering`](wrappers/misc_wrappers.md#gymnasium.wrappers.HumanRendering) or [`gymnasium.wrappers.RenderCollection`](wrappers/misc_wrappers.md#gymnasium.wrappers.RenderCollection), if `render_mode="human"` or `render_mode="rgb_array_list"` (or another `"_list"` mode) is requested but the environment only supports it through another render mode

If you want to get to the environment underneath **all** of the layers of wrappers, you can use the
[`gymnasium.Wrapper.unwrapped`](#gymnasium.Wrapper.unwrapped) attribute.
If the environment is already a bare environment, the [`gymnasium.Wrapper.unwrapped`](#gymnasium.Wrapper.unwrapped) attribute will just return itself.

```pycon
>>> wrapped_env
<RescaleAction<TimeLimit<OrderEnforcing<PassiveEnvChecker<HopperEnv<Hopper-v4>>>>>>
>>> wrapped_env.unwrapped 
<gymnasium.envs.mujoco.hopper_v4.HopperEnv object at 0x7fbb5efd0490>
```

There are three common things you might want a wrapper to do:

- Transform actions before applying them to the base environment
- Transform observations that are returned by the base environment
- Transform rewards that are returned by the base environment

Such wrappers can be easily implemented by inheriting from [`gymnasium.ActionWrapper`](wrappers/action_wrappers.md#gymnasium.ActionWrapper),
[`gymnasium.ObservationWrapper`](wrappers/observation_wrappers.md#gymnasium.ObservationWrapper), or [`gymnasium.RewardWrapper`](wrappers/reward_wrappers.md#gymnasium.RewardWrapper) and implementing the respective transformation.
If you need a wrapper to do more complicated tasks, you can inherit from the [`gymnasium.Wrapper`](#gymnasium.Wrapper) class directly.

The wrappers in this module are for single environments, [`gymnasium.Env`](env.md#gymnasium.Env). Vector environments, [`gymnasium.vector.VectorEnv`](vector.md#gymnasium.vector.VectorEnv),
have their own [vector wrappers](vector/wrappers.md) in `gymnasium.wrappers.vector` that operate on the batched data, for example, [`gymnasium.wrappers.vector.NormalizeObservation`](vector/wrappers.md#gymnasium.wrappers.vector.NormalizeObservation).
Alternatively, single-environment wrappers can be applied to each sub-environment with the `wrappers` argument of [`gymnasium.make_vec()`](registry.md#gymnasium.make_vec).

If you’d like to implement your own custom wrapper, check out [the corresponding tutorial](../../tutorials/gymnasium_basics/implementing_custom_wrappers).

### *class* gymnasium.Wrapper(env: [Env](env.md#gymnasium.Env)[WrapperObsType, WrapperActType])

Wraps a [`gymnasium.Env`](env.md#gymnasium.Env) to allow a modular transformation of the [`step()`](#gymnasium.Wrapper.step) and [`reset()`](#gymnasium.Wrapper.reset) methods.

This class is the base class of all wrappers to change the behavior of the underlying environment.
Wrappers that inherit from this class can modify the [`action_space`](#gymnasium.Wrapper.action_space), [`observation_space`](#gymnasium.Wrapper.observation_space)
and [`metadata`](#gymnasium.Wrapper.metadata) attributes, without changing the underlying environment’s attributes.
Moreover, the behavior of the [`step()`](#gymnasium.Wrapper.step) and [`reset()`](#gymnasium.Wrapper.reset) methods can be changed by these wrappers.

Some attributes ([`spec`](registry.md#gymnasium.spec), `render_mode`, [`np_random`](#gymnasium.Wrapper.np_random)) will point back to the wrapper’s environment
(i.e. to the corresponding attributes of [`env`](#gymnasium.Wrapper.env)).

#### NOTE
If you inherit from [`Wrapper`](#gymnasium.Wrapper), don’t forget to call `super().__init__(env)`

* **Parameters:**
  **env** – The environment to wrap

## Methods

#### Wrapper.step(action: WrapperActType) → tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]

Uses the [`step()`](#gymnasium.Wrapper.step) of the [`env`](#gymnasium.Wrapper.env) that can be overwritten to change the returned data.

#### Wrapper.reset(\*, seed: int | None = None, options: dict[str, Any] | None = None) → tuple[WrapperObsType, dict[str, Any]]

Uses the [`reset()`](#gymnasium.Wrapper.reset) of the [`env`](#gymnasium.Wrapper.env) that can be overwritten to change the returned data.

#### Wrapper.render() → str | ndarray[Any, Any] | tuple[ndarray[Any, Any], ndarray[Any, Any]] | list[str | ndarray[Any, Any] | tuple[ndarray[Any, Any], ndarray[Any, Any]]] | None

Uses the [`render()`](#gymnasium.Wrapper.render) of the [`env`](#gymnasium.Wrapper.env) that can be overwritten to change the returned data.

#### Wrapper.close() → None

Closes the wrapper and [`env`](#gymnasium.Wrapper.env).

#### *classmethod* Wrapper.wrapper_spec(\*\*kwargs: Any) → [WrapperSpec](registry.md#gymnasium.envs.registration.WrapperSpec)

Generates a WrapperSpec for the wrappers.

#### Wrapper.has_wrapper_attr(name: str) → bool

Checks if the given attribute is within the wrapper or its environment.

#### Wrapper.get_wrapper_attr(name: str) → Any

Gets an attribute from the wrapper and lower environments if name doesn’t exist in this object.

* **Parameters:**
  **name** – The variable name to get
* **Returns:**
  **The variable with name in wrapper or lower environments**

#### Wrapper.set_wrapper_attr(name: str, value: Any, \*, force: bool = True) → bool

Sets an attribute on this wrapper or lower environment if name is already defined.

* **Parameters:**
  * **name** – The variable name
  * **value** – The new variable value
  * **force** – Whether to create the attribute on this wrapper if it does not exists on the
    lower environment instead of raising an exception
* **Returns:**
  **If the variable has been set in this or a lower wrapper.**

## Attributes

#### Wrapper.env *: [Env](env.md#gymnasium.Env)[WrapperObsType, WrapperActType]*

The environment (one level underneath) this wrapper.

This may itself be a wrapped environment. To obtain the environment underneath all layers of wrappers, use [`gymnasium.Wrapper.unwrapped`](#gymnasium.Wrapper.unwrapped).

#### *property* Wrapper.action_space *: [Space](spaces.md#gymnasium.spaces.Space)[ActType] | [Space](spaces.md#gymnasium.spaces.Space)[WrapperActType]*

Return the [`Env`](env.md#gymnasium.Env) [`action_space`](#gymnasium.Wrapper.action_space) unless overwritten then the wrapper [`action_space`](#gymnasium.Wrapper.action_space) is used.

#### *property* Wrapper.observation_space *: [Space](spaces.md#gymnasium.spaces.Space)[ObsType] | [Space](spaces.md#gymnasium.spaces.Space)[WrapperObsType]*

Return the [`Env`](env.md#gymnasium.Env) [`observation_space`](#gymnasium.Wrapper.observation_space) unless overwritten then the wrapper [`observation_space`](#gymnasium.Wrapper.observation_space) is used.

#### *property* Wrapper.spec *: [EnvSpec](registry.md#gymnasium.envs.registration.EnvSpec) | None*

Returns the [`Env`](env.md#gymnasium.Env) [`spec`](registry.md#gymnasium.spec) attribute with the WrapperSpec if the wrapper inherits from EzPickle.

#### *property* Wrapper.metadata *: dict[str, Any]*

Returns the [`Env`](env.md#gymnasium.Env) [`metadata`](#gymnasium.Wrapper.metadata).

#### *property* Wrapper.np_random *: Generator*

Returns the [`Env`](env.md#gymnasium.Env) [`np_random`](#gymnasium.Wrapper.np_random) attribute.

#### *property* Wrapper.np_random_seed *: int*

Returns the base environment’s [`np_random_seed`](#gymnasium.Wrapper.np_random_seed).

#### *property* Wrapper.unwrapped *: [Env](env.md#gymnasium.Env)[ObsType, ActType]*

Returns the base environment of the wrapper.

This will be the bare [`gymnasium.Env`](env.md#gymnasium.Env) environment, underneath all layers of wrappers.
