# Car Racing

![image](_static/videos/box2d/car_racing.gif)

This environment is part of the [Box2D environments](../box2d.md) which contains general information about the environment.

|                   |                                        |
|-------------------|----------------------------------------|
| Action Space      | `Box([-1. 0. 0.], 1.0, (3,), float32)` |
| Observation Space | `Box(0, 255, (96, 96, 3), uint8)`      |
| Max Episode Steps | `1000`                                 |
| Reward Threshold  | `900`                                  |
| import            | `gymnasium.make("CarRacing-v3")`       |

## Description

The easiest control task to learn from pixels - a top-down
racing environment. The generated track is random every episode.

Some indicators are shown at the bottom of the window along with the
state RGB buffer. From left to right: true speed, four ABS sensors,
steering wheel position, and gyroscope.
To play yourself (it’s rather fast for humans), type:

```shell
python gymnasium/envs/box2d/car_racing.py
```

Remember: it’s a powerful rear-wheel drive car - don’t press the accelerator
and turn at the same time.

## Action Space

If continuous (the default), the action is a `Box([-1, 0, 0], [1, 1, 1], (3,), float32)`:

- 0: steering, -1 is full left, +1 is full right
- 1: gas, in `[0, 1]`
- 2: braking, in `[0, 1]`

If discrete, the action space is `Discrete(5)`:

- 0: do nothing
- 1: steer right
- 2: steer left
- 3: gas
- 4: brake

## Observation Space

A top-down 96x96 RGB image of the car and race track, as a `Box(0, 255, (96, 96, 3), uint8)`.

## Rewards

The reward is -0.1 every frame and +1000/N for the first visit to each track tile, where N is the total number
of tiles in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1\*732 = 926.8 points.

## Starting State

The car starts at rest in the center of the road.

## Episode End

The episode terminates when all the tiles are visited, with `info["lap_finished"] = True`.
The car can also go outside the playfield - that is, far off the track, in which case the reward for that
step is -100 and the episode terminates with `info["lap_finished"] = False`.

The episode is truncated after 1000 steps (the `max_episode_steps` of `CarRacing-v3`).

## Information

`step()` returns an empty `info` dict, except on termination when it contains `lap_finished` (see above).
`reset()` returns an empty `info` dict.

## Arguments

```python
>>> import gymnasium as gym
>>> env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>

```

* `lap_complete_percent=0.95` is intended to dictate the fraction of tiles that must be visited by
  the agent before a lap is considered complete, checked when the car first visits the start tile.
  However, the start tile is already visited when the episode starts, so this check does not currently
  trigger and a lap is only complete once every tile has been visited.
* `domain_randomize=False` enables the domain randomized variant of the environment.
  In this scenario, the background and track colours are different on every reset.
* `continuous=True` specifies if the agent has continuous (true) or discrete (false) actions.
  See action space section for a description of each.
* `verbose=False` prints track generation information on reset if `True`.
* `render_mode` supports `"human"`, `"rgb_array"` (a 600x400 image) and `"state_pixels"`
  (a 96x96 image, the same as the observation).

## Reset Arguments

Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
`domain_randomize` must be `True` on init for this argument to work.

```python
>>> import gymnasium as gym
>>> env = gym.make("CarRacing-v3", domain_randomize=True)

# normal reset, this changes the colour scheme by default
>>> obs, _ = env.reset()

# reset with colour scheme change
>>> randomize_obs, _ = env.reset(options={"randomize": True})

# reset with no colour scheme change
>>> non_random_obs, _ = env.reset(options={"randomize": False})

```

## Version History

- v3: Change truncation to termination when finishing the lap (1.0.0)
- v2: Remove `CarRacingDiscrete-v1`, use `continuous=False` instead (0.26.0)
- v1: Change track completion logic and add domain randomization (0.24.0)
- v0: Original version

## References

- Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

## Credits

Created by Oleg Klimov
