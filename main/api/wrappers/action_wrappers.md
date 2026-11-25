# Action Wrappers

## Base class

### *class* gymnasium.ActionWrapper(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType])

Superclass of wrappers that can modify the action before `step()`.

If you would like to apply a function to the action before passing it to the base environment,
you can simply inherit from [`ActionWrapper`](#gymnasium.ActionWrapper) and overwrite the method [`action()`](#gymnasium.ActionWrapper.action) to implement
that transformation. The transformation defined in that method must take values in the base environment’s
action space. However, its domain might differ from the original action space.
In that case, you need to specify the new action space of the wrapper by setting `action_space` in
the `__init__()` method of your wrapper.

Among others, Gymnasium provides the action wrappers [`gymnasium.wrappers.ClipAction`](#gymnasium.wrappers.ClipAction) and
[`gymnasium.wrappers.RescaleAction`](#gymnasium.wrappers.RescaleAction) for clipping and rescaling actions.

* **Parameters:**
  **env** – Environment to be wrapped.

#### action(action: WrapperActType) → ActType

Returns a modified action before `step()` is called.

* **Parameters:**
  **action** – The original `step()` actions
* **Returns:**
  **The modified actions**

## Implemented wrappers

### *class* gymnasium.wrappers.TransformAction(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], func: Callable[[WrapperActType], ActType], action_space: [Space](../spaces.md#gymnasium.spaces.Space)[WrapperActType] | None)

Applies a function to the `action` before passing the modified value to the environment `step` function.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.TransformAction`](../vector/wrappers.md#gymnasium.wrappers.vector.TransformAction).

### Example

```pycon
>>> import numpy as np
>>> import gymnasium as gym
>>> env = gym.make("MountainCarContinuous-v0")
>>> _ = env.reset(seed=123)
>>> obs, *_= env.step(np.array([0.0, 1.0]))
>>> obs
array([-4.6397772e-01, -4.4808415e-04], dtype=float32)
>>> env = gym.make("MountainCarContinuous-v0")
>>> env = TransformAction(env, lambda a: 0.5 * a + 0.1, env.action_space)
>>> _ = env.reset(seed=123)
>>> obs, *_= env.step(np.array([0.0, 1.0]))
>>> obs
array([-4.6382770e-01, -2.9808417e-04], dtype=float32)
```

Change logs:
: * v1.0.0 - Initially added

* **Parameters:**
  * **env** – The environment to wrap
  * **func** – Function to apply to the `step()`’s `action`
  * **action_space** – The updated action space of the wrapper given the function.

### *class* gymnasium.wrappers.ClipAction(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType])

Clips the `action` pass to `step` to be within the environment’s action_space.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.ClipAction`](../vector/wrappers.md#gymnasium.wrappers.vector.ClipAction).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import ClipAction
>>> import numpy as np
>>> env = gym.make("Hopper-v4", disable_env_checker=True)
>>> env = ClipAction(env)
>>> env.action_space
Box(-inf, inf, (3,), float32)
>>> _ = env.reset(seed=42)
>>> _ = env.step(np.array([5.0, -2.0, 0.0], dtype=np.float32))
... # Executes the action np.array([1.0, -1.0, 0]) in the base environment
```

Change logs:
: * v0.12.6 - Initially added
  * v1.0.0 - Action space is updated to infinite bounds as is technically correct

* **Parameters:**
  **env** – The environment to wrap

### *class* gymnasium.wrappers.RescaleAction(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], min_action: floating | integer | ndarray, max_action: floating | integer | ndarray)

Affinely (linearly) rescales a `Box` action space of the environment to within the range of `[min_action, max_action]`.

The base environment `env` must have an action space of type `spaces.Box`. If `min_action`
or `max_action` are numpy arrays, the shape must match the shape of the environment’s action space.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.RescaleAction`](../vector/wrappers.md#gymnasium.wrappers.vector.RescaleAction).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import RescaleAction
>>> import numpy as np
>>> env = gym.make("Hopper-v4", disable_env_checker=True)
>>> _ = env.reset(seed=42)
>>> obs, _, _, _, _ = env.step(np.array([1, 1, 1], dtype=np.float32))
>>> _ = env.reset(seed=42)
>>> min_action = -0.5
>>> max_action = np.array([0.0, 0.5, 0.75], dtype=np.float32)
>>> wrapped_env = RescaleAction(env, min_action=min_action, max_action=max_action)
>>> wrapped_env_obs, _, _, _, _ = wrapped_env.step(max_action)
>>> np.all(obs == wrapped_env_obs)
np.True_
```

Change logs:
: * v0.15.4 - Initially added

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The environment to wrap
  * **min_action** (*float* *,* *int* *or* *np.ndarray*) – The min values for each action. This may be a numpy array or a scalar.
  * **max_action** (*float* *,* *int* *or* *np.ndarray*) – The max values for each action. This may be a numpy array or a scalar.
* **Raises:**
  [**InvalidBound**](../utils.md#gymnasium.error.InvalidBound) – If a component of `min_action` equals the matching component of `max_action`.

### *class* gymnasium.wrappers.StickyAction(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], repeat_action_probability: float, repeat_action_duration: int | tuple[int, int] = 1)

Adds a probability that the action is repeated for the same `step` function.

This wrapper follows the implementation proposed by [Machado et al., 2018](https://arxiv.org/pdf/1709.06009.pdf)
in Section 5.2 on page 12, and adds the possibility to repeat the action for
more than one step.

No vector version of the wrapper exists.

### Example

```pycon
>>> import gymnasium as gym
>>> env = gym.make("CartPole-v1")
>>> env = StickyAction(env, repeat_action_probability=0.9)
>>> env.reset(seed=123)
(array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})
>>> env.step(1)
(array([ 0.01734283,  0.15089367, -0.02859527, -0.33293587], dtype=float32), 1.0, False, False, {})
>>> env.step(0)
(array([ 0.0203607 ,  0.34641072, -0.03525399, -0.6344974 ], dtype=float32), 1.0, False, False, {})
>>> env.step(1)
(array([ 0.02728892,  0.5420062 , -0.04794393, -0.9380709 ], dtype=float32), 1.0, False, False, {})
>>> env.step(0)
(array([ 0.03812904,  0.34756234, -0.06670535, -0.6608303 ], dtype=float32), 1.0, False, False, {})
```

Change logs:
: * v1.0.0 - Initially added
  * v1.1.0 - Add repeat_action_duration argument for dynamic number of sticky actions

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – the wrapped environment,
  * **repeat_action_probability** (*int* *|* *float*) – a probability of repeating the old action,
  * **repeat_action_duration** (*int* *|* *tuple* *[**int* *,* *int* *]*) – the number of steps
    the action is repeated. It can be either an int (for deterministic
    repeats) or a tuple[int, int] for a range of stochastic number of repeats.

### *class* gymnasium.wrappers.RepeatAction(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], num_repeats: int)

Repeats the given action for `num_repeats` steps, accumulating the reward.

This is useful for environments where the agent does not need to make
a decision at every time step. By repeating actions, the effective
decision frequency is reduced, which can speed up training and make
exploration more efficient.

Unlike [`StickyAction`](#gymnasium.wrappers.StickyAction), which *stochastically* replaces the agent’s
action with the *previous* action, `ActionRepeat` *deterministically*
executes the *current* action multiple times and returns the accumulated
reward together with the final observation. If the episode terminates or
is truncated during repetition, the loop stops early.

No vector version of the wrapper exists.

### Example

```pycon
>>> import gymnasium as gym
>>> env = gym.make("CartPole-v1")
>>> env = RepeatAction(env, num_repeats=4)
>>> obs, info = env.reset(seed=123)
>>> obs, reward, terminated, truncated, info = env.step(1)
>>> reward  # sum of 4 inner-step rewards
4.0
```

Change logs:
: * v1.3.0 - Initially added

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The wrapped environment.
  * **num_repeats** (*int*) – The number of times to repeat each action.

### *class* gymnasium.wrappers.DiscretizeAction(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], bins: int | tuple[int, ...], multidiscrete: bool = False)

Uniformly discretizes a continuous Box action space into a single Discrete space.

Example 1 - Discretize Pendulum action space:
: ```pycon
  >>> env = gym.make("Pendulum-v1")
  >>> env.action_space
  Box(-2.0, 2.0, (1,), float32)
  >>> obs, _ = env.reset(seed=42)
  >>> obs, *_ = env.step([-0.6])
  >>> obs
  array([-0.17606162,  0.9843792 ,  0.5292768 ], dtype=float32)
  >>> env = DiscretizeAction(env, bins=10)
  >>> env.action_space
  Discrete(10)
  >>> obs, _ = env.reset(seed=42)
  >>> obs, *_ = env.step(3)
  >>> obs
  array([-0.17606162,  0.9843792 ,  0.5292768 ], dtype=float32)
  ```

Example 2 - Discretize Reacher action space:
: ```pycon
  >>> env = gym.make("Reacher-v5")
  >>> env.action_space
  Box(-1.0, 1.0, (2,), float32)
  >>> obs, _ = env.reset(seed=42)
  >>> obs, *_ = env.step([-0.3, -0.5])
  >>> obs
  array([ 0.99908342,  0.99948506,  0.04280567, -0.03208766,  0.10445588,
          0.11442572, -1.18958125, -1.97979484,  0.10544613, -0.10896536])
  >>> env = DiscretizeAction(env, bins=10)
  >>> env.action_space
  Discrete(100)
  >>> obs, _ = env.reset(seed=42)
  >>> obs, *_ = env.step(32)
  >>> obs
  array([ 0.99908342,  0.99948506,  0.04280567, -0.03208766,  0.10445588,
          0.11442572, -1.18958118, -1.97979484,  0.10544613, -0.10896536])
  ```

Example 2 - Discretize Reacher action space with MultiDiscrete:
: ```pycon
  >>> env = gym.make("Reacher-v5")
  >>> env.action_space
  Box(-1.0, 1.0, (2,), float32)
  >>> obs, _ = env.reset(seed=42)
  >>> obs, *_ = env.step([-0.3, -0.5])
  >>> obs
  array([ 0.99908342,  0.99948506,  0.04280567, -0.03208766,  0.10445588,
          0.11442572, -1.18958125, -1.97979484,  0.10544613, -0.10896536])
  >>> env = DiscretizeAction(env, bins=10, multidiscrete=True)
  >>> env.action_space
  MultiDiscrete([10 10])
  >>> obs, _ = env.reset(seed=42)
  >>> obs, *_ = env.step([3, 2])
  >>> obs
  array([ 0.99908342,  0.99948506,  0.04280567, -0.03208766,  0.10445588,
          0.11442572, -1.18958118, -1.97979484,  0.10544613, -0.10896536])
  ```

* **Parameters:**
  * **env** – The environment to wrap.
  * **bins** – int or tuple of ints (number of bins per dimension).
  * **multidiscrete** – If True, use MultiDiscrete action space instead of flattening to Discrete.
