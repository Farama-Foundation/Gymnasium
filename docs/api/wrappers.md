---
title: Wrapper
---

# Wrappers

```{toctree}
:hidden:
wrappers/general_wrappers
wrappers/action_wrappers
wrappers/observation_wrappers
wrappers/reward_wrappers
```

## gymnasium.Wrapper

```{eval-rst}
.. autoclass:: gymnasium.Wrapper
```

### Methods
```{eval-rst}
.. autofunction:: gymnasium.Wrapper.step
.. autofunction:: gymnasium.Wrapper.reset
.. autofunction:: gymnasium.Wrapper.close
```

### Attributes

```{eval-rst}
.. autoproperty:: gymnasium.Wrapper.action_space
.. autoproperty:: gymnasium.Wrapper.observation_space
.. autoproperty:: gymnasium.Wrapper.reward_range
.. autoproperty:: gymnasium.Wrapper.spec
.. autoproperty:: gymnasium.Wrapper.metadata
.. autoproperty:: gymnasium.Wrapper.np_random
.. autoproperty:: gymnasium.Wrapper.unwrapped
```

## Gymnasium Wrappers

| Name                      | Type                 | Description                                                                                                                                                                            |
|---------------------------|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `AtariPreprocessing`      | `Wrapper`            | Implements the common preprocessing applied tp Atari environments                                                                                                                      |
| `AutoResetWrapper`        | `Wrapper`            | The wrapped environment will automatically reset when the terminated or truncated  state is reached.                                                                                   |
| `ClipAction`              | `ActionWrapper`      | Clip the continuous action to the valid bound specified by the environment's `action_space`                                                                                            |
| `FilterObservation`       | `ObservationWrapper` | Filters a dictionary observation spaces to only requested keys                                                                                                                         | 
| `FlattenObservation`      | `ObservationWrapper` | An Observation wrapper that flattens the observation                                                                                                                                   |
| `FrameStack`              | `ObservationWrapper` | AnObservation wrapper that stacks the observations in a rolling manner.                                                                                                                |
| `GrayScaleObservation`    | `ObservationWrapper` | Convert the image observation from RGB to gray scale.                                                                                                                                  |
| `NormalizeReward`         | `Wrapper`            | This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.                                                                              |
| `NormalizeObservation`    | `Wrapper`            | This wrapper will normalize observations s.t. each coordinate is centered with unit variance.                                                                                          |
| `OrderEnforcing`          | `Wrapper`            | This will produce an error if `step` or `render` is called before `reset`                                                                                                              |
| `PixelObservationWrapper` | `ObservationWrapper` | Augment observations by pixel values obtained via `render` that can be added to or replaces the environments observation.                                                              |
| `RecordEpisodeStatistics` | `Wrapper`            | This will keep track of cumulative rewards and episode lengths returning them at the end.                                                                                              |
| `RecordVideo`             | `Wrapper`            | This wrapper will record videos of rollouts.                                                                                                                                           |
| `RescaleAction`           | `ActionWrapper`      | Rescales the continuous action space of the environment to a range \[`min_action`, `max_action`], where `min_action` and `max_action` are numpy arrays or floats.                      |
| `ResizeObservation`       | `ObservationWrapper` | This wrapper works on environments with image observations (or more generally observations of shape AxBxC) and resizes the observation to the shape given by the tuple `shape`.        |
| `TimeAwareObservation`    | `ObservationWrapper` | Augment the observation with current time step in the trajectory (by appending it to the observation).                                                                                 |
| `TimeLimit`               | `Wrapper`            | This wrapper will emit a truncated signal if the specified number of steps is exceeded in an episode.                                                                                  |
| `TransformObservation`    | `ObservationWrapper` | This wrapper will apply function to observations                                                                                                                                       |
| `TransformReward`         | `RewardWrapper`      | This wrapper will apply function to rewards                                                                                                                                            |
| `VectorListInfo`          | `Wrapper`            | This wrapper will convert the info of a vectorized environment from the `dict` format to a `list` of dictionaries where the _i-th_ dictionary contains info of the _i-th_ environment. |


## Implementing a custom wrapper

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
import gymnasium as gym

class ReacherRewardWrapper(gym.Wrapper):
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