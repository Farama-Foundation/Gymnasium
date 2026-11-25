# Reward Wrappers

### *class* gymnasium.RewardWrapper(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType])

Superclass of wrappers that can modify the returning reward from a step.

If you would like to apply a function to the reward that is returned by the base environment before
passing it to learning code, you can simply inherit from [`RewardWrapper`](#gymnasium.RewardWrapper) and overwrite the method
[`reward()`](#gymnasium.RewardWrapper.reward) to implement that transformation.

* **Parameters:**
  **env** – Environment to be wrapped.

#### reward(reward: SupportsFloat) → SupportsFloat

Returns a modified environment `reward`.

* **Parameters:**
  **reward** – The `env` `step()` reward
* **Returns:**
  **The modified \`reward\`**

## Implemented wrappers

### *class* gymnasium.wrappers.TransformReward(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], func: Callable[[SupportsFloat], SupportsFloat])

Applies a function to the `reward` received from the environment’s `step`.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.TransformReward`](../vector/wrappers.md#gymnasium.wrappers.vector.TransformReward).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import TransformReward
>>> env = gym.make("CartPole-v1")
>>> env = TransformReward(env, lambda r: 2 * r + 1)
>>> _ = env.reset()
>>> _, rew, _, _, _ = env.step(0)
>>> rew
3.0
```

Change logs:
: * v0.15.0 - Initially added

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The environment to wrap
  * **func** – (Callable): The function to apply to reward

### *class* gymnasium.wrappers.NormalizeReward(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], gamma: float = 0.99, epsilon: float = 1e-8)

Normalizes immediate rewards such that their exponential moving average has an approximately fixed variance.

The property \_update_running_mean allows to freeze/continue the running mean calculation of the reward
statistics. If True (default), the RunningMeanStd will get updated every time self.normalize() is called.
If False, the calculated statistics are used but not updated anymore; this may be used during evaluation.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.NormalizeReward`](../vector/wrappers.md#gymnasium.wrappers.vector.NormalizeReward).

#### NOTE
In v0.27, NormalizeReward was updated as the forward discounted reward estimate was incorrectly computed in Gym v0.25+.
For more detail, read [#3154]([https://github.com/openai/gym/pull/3152](https://github.com/openai/gym/pull/3152)).

#### NOTE
The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
instantiated or the policy was changed recently.

Example without the normalize reward wrapper:
: ```pycon
  >>> import numpy as np
  >>> import gymnasium as gym
  >>> env = gym.make("MountainCarContinuous-v0")
  >>> _ = env.reset(seed=123)
  >>> _ = env.action_space.seed(123)
  >>> episode_rewards = []
  >>> terminated, truncated = False, False
  >>> while not (terminated or truncated):
  ...     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
  ...     episode_rewards.append(reward)
  ...
  >>> env.close()
  >>> np.var(episode_rewards)
  np.float64(0.0008876301247721108)
  ```

Example with the normalize reward wrapper:
: ```pycon
  >>> import numpy as np
  >>> import gymnasium as gym
  >>> env = gym.make("MountainCarContinuous-v0")
  >>> env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
  >>> _ = env.reset(seed=123)
  >>> _ = env.action_space.seed(123)
  >>> episode_rewards = []
  >>> terminated, truncated = False, False
  >>> while not (terminated or truncated):
  ...     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
  ...     episode_rewards.append(reward)
  ...
  >>> env.close()
  >>> np.var(episode_rewards)
  np.float64(0.010162116476634746)
  ```

Change logs:
: * v0.21.0 - Initially added
  * v1.0.0 - Add update_running_mean attribute to allow disabling of updating the running mean / standard

* **Parameters:**
  * **env** (*env*) – The environment to apply the wrapper
  * **epsilon** (*float*) – A stability parameter
  * **gamma** (*float*) – The discount factor that is used in the exponential moving average.
* **Raises:**
  [**InvalidBound**](../utils.md#gymnasium.error.InvalidBound) – If `gamma` is outside `[0, 1]`, or if `epsilon` is not finite and strictly positive.

### *class* gymnasium.wrappers.ClipReward(env: [Env](../env.md#gymnasium.Env)[ObsType, ActType], min_reward: float | ndarray | None = None, max_reward: float | ndarray | None = None)

Clips the rewards for an environment between an upper and lower bound.

A vector version of the wrapper exists [`gymnasium.wrappers.vector.ClipReward`](../vector/wrappers.md#gymnasium.wrappers.vector.ClipReward).

### Example

```pycon
>>> import gymnasium as gym
>>> from gymnasium.wrappers import ClipReward
>>> env = gym.make("CartPole-v1")
>>> env = ClipReward(env, 0, 0.5)
>>> _ = env.reset()
>>> _, rew, _, _, _ = env.step(1)
>>> rew
np.float64(0.5)
```

Change logs:
: * v1.0.0 - Initially added

* **Parameters:**
  * **env** ([*Env*](../env.md#gymnasium.Env)) – The environment to wrap
  * **min_reward** (*Union* *[**float* *,* *np.ndarray* *]*) – lower bound to apply
  * **max_reward** (*Union* *[**float* *,* *np.ndarray* *]*) – higher bound to apply
