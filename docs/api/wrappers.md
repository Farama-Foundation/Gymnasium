---
title: Wrappers
lastpage:
---

# Wrappers

Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly.
Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular. Wrappers can 
also be chained to combine their effects. Most environments that are generated via `gymnasium.make` will already be wrapped by default.

In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along
with (possibly optional) parameters to the wrapper's constructor:
```python
>>> import gymnasium
>>> from gymnasium.wrappers import RescaleAction
>>> base_env = gymnasium.make("BipedalWalker-v3")
>>> base_env.action_space
Box([-1. -1. -1. -1.], [1. 1. 1. 1.], (4,), float32)
>>> wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
>>> wrapped_env.action_space
Box([0. 0. 0. 0.], [1. 1. 1. 1.], (4,), float32)
```
You can access the environment underneath the **first** wrapper by using
the `.env` attribute:

```python
>>> wrapped_env
<RescaleAction<TimeLimit<OrderEnforcing<BipedalWalker<BipedalWalker-v3>>>>>
>>> wrapped_env.env
<TimeLimit<OrderEnforcing<BipedalWalker<BipedalWalker-v3>>>>
```

If you want to get to the environment underneath **all** of the layers of wrappers, 
you can use the `.unwrapped` attribute. 
If the environment is already a bare environment, the `.unwrapped` attribute will just return itself.

```python
>>> wrapped_env
<RescaleAction<TimeLimit<OrderEnforcing<BipedalWalker<BipedalWalker-v3>>>>>
>>> wrapped_env.unwrapped
<gymnasium.envs.box2d.bipedal_walker.BipedalWalker object at 0x7f87d70712d0>
```

There are three common things you might want a wrapper to do:

- Transform actions before applying them to the base environment
- Transform observations that are returned by the base environment
- Transform rewards that are returned by the base environment

Such wrappers can be easily implemented by inheriting from `ActionWrapper`, `ObservationWrapper`, or `RewardWrapper` and implementing the
respective transformation. If you need a wrapper to do more complicated tasks, you can inherit from the `Wrapper` class directly.
The code that is presented in the following sections can also be found in 
the [gym-examples](https://github.com/Farama-Foundation/gym-examples) repository

## ActionWrapper
If you would like to apply a function to the action before passing it to the base environment,
you can simply inherit from `ActionWrapper` and overwrite the method `action` to implement that transformation.
The transformation defined in that method must take values in the base environment's action space.
However, its domain might differ from the original action space. In that case, you need to specify the new
action space of the wrapper by setting `self.action_space` in the `__init__` method of your wrapper.

Let's say you have an environment with action space of type `Box`, but you would
only like to use a finite subset of actions. Then, you might want to implement the following wrapper

```python
class DiscreteActions(gymnasium.ActionWrapper):
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))
    
    def action(self, act):
        return self.disc_to_cont[act]

if __name__ == "__main__":
    env = gymnasium.make("LunarLanderContinuous-v2")
    wrapped_env = DiscreteActions(env, [np.array([1,0]), np.array([-1,0]),
                                        np.array([0,1]), np.array([0,-1])])
    print(wrapped_env.action_space)         #Discrete(4)
```

Among others, Gymnasium provides the action wrappers `ClipAction` and `RescaleAction`.

## ObservationWrapper
If you would like to apply a function to the observation that is returned by the base environment before passing
it to learning code, you can simply inherit from `ObservationWrapper` and overwrite the method `observation` to 
implement that transformation. The transformation defined in that method must be defined on the base environment's
observation space. However, it may take values in a different space. In that case, you need to specify the new
observation space of the wrapper by setting `self.observation_space` in the `__init__` method of your wrapper.

For example, you might have a 2D navigation task where the environment returns dictionaries as observations with keys `"agent_position"`
and `"target_position"`. A common thing to do might be to throw away some degrees of freedom and only consider
the position of the target relative to the agent, i.e. `observation["target_position"] - observation["agent_position"]`. 
For this, you could implement an observation wrapper like this:

```python
class RelativePosition(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

    def observation(self, obs):
        return obs["target"] - obs["agent"]
```

Among others, Gymnasium provides the observation wrapper `TimeAwareObservation`, which adds information about the index of the timestep 
to the observation.

## RewardWrapper
If you would like to apply a function to the reward that is returned by the base environment before passing
it to learning code, you can simply inherit from `RewardWrapper` and overwrite the method `reward` to 
implement that transformation. This transformation might change the reward range; to specify the reward range of 
your wrapper, you can simply define `self.reward_range` in `__init__`.

Let us look at an example: Sometimes (especially when we do not have control over the reward because it is intrinsic), we want to clip the reward
to a range to gain some numerical stability. To do that, we could, for instance, implement the following wrapper:

```python
class ClipReward(gymnasium.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)
```

## AutoResetWrapper

Some users may want a wrapper which will automatically reset its wrapped environment when its wrapped environment reaches the done state. An advantage of this environment is that it will never produce undefined behavior as standard gymnasium environments do when stepping beyond the done state. 

When calling step causes `self.env.step()` to return `(terminated or truncated)=True`,
`self.env.reset()` is called,
and the return format of `self.step()` is as follows:

```python
new_obs, final_reward, final_terminated, final_truncated, info
```

`new_obs` is the first observation after calling `self.env.reset()`,

`final_reward` is the reward after calling `self.env.step()`,
prior to calling `self.env.reset()`

The expression `(final_terminated or final_truncated)` is always `True`

`info` is a dict containing all the keys from the info dict returned by
the call to `self.env.reset()`, with additional keys `final_observation`
containing the observation returned by the last call to `self.env.step()`
and `final_info` containing the info dict returned by the last call
to `self.env.step()`.

If `(terminated or truncated)` is not true when `self.env.step()` is called, `self.step()` returns

```python
obs, reward, terminated, truncated, info
```
as normal.


The AutoResetWrapper is not applied by default when calling `gymnasium.make()`, but can be applied by setting the optional `autoreset` argument to `True`:

```python
    env = gymnasium.make("CartPole-v1", autoreset=True)
```

The AutoResetWrapper can also be applied using its constructor:
```python
    env = gymnasium.make("CartPole-v1")
    env = AutoResetWrapper(env)
```


```{note}
When using the  AutoResetWrapper to collect rollouts, note
that the when `self.env.step()` returns `done`, a
new observation from after calling `self.env.reset()` is returned
by `self.step()` alongside the terminal reward and done state from the
previous episode . If you need the terminal state from the previous
episode, you need to retrieve it via the the `final_observation` key
in the info dict. Make sure you know what you're doing if you
use this wrapper!
```


## General Wrappers

Sometimes you might need to implement a wrapper that does some more complicated modifications (e.g. modify the
reward based on data in `info` or change the rendering behavior). 
Such wrappers can be implemented by inheriting from `Wrapper`. 

- You can set a new action or observation space by defining `self.action_space` or `self.observation_space` in `__init__`, respectively
- You can set new metadata and reward range by defining `self.metadata` and `self.reward_range` in `__init__`, respectively
- You can override `step`, `render`, `close` etc. If you do this, you can access the environment that was passed
to your wrapper (which *still* might be wrapped in some other wrapper) by accessing the attribute `self.env`.

Let's also take a look at an example for this case. Most MuJoCo environments return a reward that consists
of different terms: For instance, there might be a term that rewards the agent for completing the task and one term that
penalizes large actions (i.e. energy usage). Usually, you can pass weight parameters for those terms during
initialization of the environment. However, *Reacher* does not allow you to do this! Nevertheless, all individual terms
of the reward are returned in `info`, so let us build a wrapper for Reacher that allows us to weight those terms:

```python
class ReacherRewardWrapper(gymnasium.Wrapper):
    def __init__(self, env, reward_dist_weight, reward_ctrl_weight):
        super().__init__(env)
        self.reward_dist_weight = reward_dist_weight
        self.reward_ctrl_weight = reward_ctrl_weight

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = (
            self.reward_dist_weight * info["reward_dist"]
            + self.reward_ctrl_weight * info["reward_ctrl"]
        )
        return obs, reward, terminated, truncated, info
```

```{note}
It is *not* sufficient to use a `RewardWrapper` in this case!
```

## Available Wrappers

| Name                      | Type                     | Arguments                                                                                                                                                                                                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|---------------------------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `AtariPreprocessing`      | `gymnasium.Wrapper`            | `env: gymnasium.Env`, `noop_max: int = 30`, `frame_skip: int = 4`, `screen_size: int = 84`, `terminal_on_life_loss: bool = False`, `grayscale_obs: bool = True`, `grayscale_newaxis: bool = False`, `scale_obs: bool = False`  | Implements the best practices from Machado et al. (2018), "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents" but will be deprecated soon.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `AutoResetWrapper`          | `gymnasium.Wrapper`            | `env`                                                                                                                                                                                                                    | The wrapped environment will automatically reset when the done state is reached. Make sure you read the documentation before using this wrapper!|
| `ClipAction`              | `gymnasium.ActionWrapper`      | `env`                                                                                                                                                                                                                    | Clip the continuous action to the valid bound specified by the environment's `action_space`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `FilterObservation`       | `gymnasium.ObservationWrapper` | `env`, `filter_keys=None`                                                                                                                                                                                                | If you have an environment that returns dictionaries as observations, but you would like to only keep a subset of the entries, you can use this wrapper. `filter_keys` should be an iterable that contains the keys that are kept in the new observation. If it is `None`, all keys will be kept and the wrapper has no effect.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 
| `FlattenObservation`      | `gymnasium.ObservationWrapper` | `env`                                                                                                                                                                                                                    | Observation wrapper that flattens the observation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `FrameStack`              | `gymnasium.ObservationWrapper` | `env`, `num_stack`, `lz4_compress=False`                                                                                                                                                                                 | Observation wrapper that stacks the observations in a rolling manner. For example, if the number of stacks is 4, then the returned observation contains the most recent 4 observations. Observations will be objects of type `LazyFrames`. This object can be cast to a numpy array via `np.asarray(obs)`. You can also access single frames or slices via the usual `__getitem__` syntax. If `lz4_compress` is set to true, the `LazyFrames` object will compress the frames internally (losslessly). The first observation (i.e. the one returned by `reset`) will consist of `num_stack` repitions of the first frame.                                                                                                                                                                                                                                                                                                      |
| `GrayScaleObservation`    | `gymnasium.ObservationWrapper` | `env`, `keep_dim=False`                                                                                                                                                                                                  | Convert the image observation from RGB to gray scale. By default, the resulting observation will be 2-dimensional. If `keep_dim` is set to true, a singleton dimension will be added (i.e. the observations are of shape AxBx1).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `NormalizeReward`         | `gymnasium.Wrapper`            | `env`, `gamma=0.99`, `epsilon=1e-8`                                                                                                                                                                                      | This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance. `epsilon` is a stability parameter and `gamma` is the discount factor that is used in the exponential moving average. The exponential moving average will have variance `(1 - gamma)**2`. The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly instantiated or the policy was changed recently.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `NormalizeObservation`    | `gymnasium.Wrapper`            | `env`, `epsilon=1e-8`                                                                                                                                                                                                    | This wrapper will normalize observations s.t. each coordinate is centered with unit variance. The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was newly instantiated or the policy was changed recently. `epsilon` is a stability parameter that is used when scaling the observations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `OrderEnforcing`          | `gymnasium.Wrapper`            | `env`                                                                                                                                                                                                                    | This will produce an error if `step` is called before an initial `reset`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `PixelObservationWrapper` | `gymnasium.ObservationWrapper` | `env`, `pixels_only=True`, `render_kwargs=None`, `pixel_keys=("pixels",)`                                                                                                                                                | Augment observations by pixel values obtained via `render`. You can specify whether the original observations should be discarded entirely or be augmented by setting `pixels_only`. Also, you can provide keyword arguments for `render`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `RecordEpisodeStatistics` | `gymnasium.Wrapper`            | `env`, `deque_size=100`                                                                                                                                                                                                  | This will keep track of cumulative rewards and episode lengths. At the end of an episode, the statistics of the episode will be added to `info`. Moreover, the rewards and episode lengths are stored in buffers that can be accessed via `wrapped_env.return_queue` and `wrapped_env.length_queue` respectively. The size of these buffers can be set via `deque_size`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `RecordVideo`             | `gymnasium.Wrapper`            | `env`, `video_folder: str`, `episode_trigger: Callable[[int], bool] = None`, `step_trigger: Callable[[int], bool] = None`, `video_length: int = 0`, `name_prefix: str = "rl-video"`                                      | This wrapper will record videos of rollouts. The results will be saved in the folder specified via `video_folder`. You can specify a prefix for the filenames via `name_prefix`. Usually, you only want to record the environment intermittently, say every hundreth episode. To allow this, you can pass `episode_trigger` or `step_trigger`. At most one of these should be passed. These functions will accept an episode index or step index, respectively. They should return a boolean that indicates whether a recording should be started at this point. If neither `episode_trigger`, nor `step_trigger` is passed, a default `episode_trigger` will be used. By default, the recording will be stopped once a done signal has been emitted by the environment. However, you can also create recordings of fixed length (possibly spanning several episodes) by passing a strictly positive value for `video_length`. |
| `RescaleAction`           | `gymnasium.ActionWrapper`      | `env`, `min_action`, `max_action`                                                                                                                                                                                        | Rescales the continuous action space of the environment to a range \[`min_action`, `max_action`], where `min_action` and `max_action` are numpy arrays or floats.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `ResizeObservation`       | `gymnasium.ObservationWrapper` | `env`, `shape`                                                                                                                                                                                                           | This wrapper works on environments with image observations (or more generally observations of shape AxBxC) and resizes the observation to the shape given by the tuple `shape`. The argument `shape` may also be an integer. In that case, the observation is scaled to a square of sidelength `shape`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `TimeAwareObservation`    | `gymnasium.ObservationWrapper` | `env`                                                                                                                                                                                                                    | Augment the observation with current time step in the trajectory (by appending it to the observation). This can be useful to ensure that things stay Markov. Currently it only works with one-dimensional observation spaces.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `TimeLimit`               | `gymnasium.Wrapper`            | `env`, `max_episode_steps=None`                                                                                                                                                                                          | Probably the most useful wrapper in Gymnasium. This wrapper will emit a done signal if the specified number of steps is exceeded in an episode. In order to be able to distinguish termination and truncation, you need to check `info`. If it does not contain the key `"TimeLimit.truncated"`, the environment did not reach the timelimit. Otherwise, `info["TimeLimit.truncated"]` will be true if the episode was terminated because of the time limit.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `TransformObservation`    | `gymnasium.ObservationWrapper` | `env`, `f`                                                                                                                                                                                                               | This wrapper will apply `f` to observations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `TransformReward`         | `gymnasium.RewardWrapper`      | `env`, `f`                                                                                                                                                                                                               | This wrapper will apply `f` to rewards                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|  `VectorListInfo` |  `gymnasium.Wrapper` | `env` | This wrapper will convert the info of a vectorized environment from the `dict` format to a `list` of dictionaries where the _i-th_ dictionary contains info of the _i-th_ environment. If using other wrappers that perform operation on info like `RecordEpisodeStatistics`, this need to be the outermost wrapper. |
