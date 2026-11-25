# Bipedal Walker

![image](_static/videos/box2d/bipedal_walker.gif)

This environment is part of the [Box2D environments](../box2d.md) which contains general information about the environment.

|                   |                                                                                                                                                                                                                                                                             |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Action Space      | `Box(-1.0, 1.0, (4,), float32)`                                                                                                                                                                                                                                             |
| Observation Space | `Box([-3.1415927 -5. -5. -5. -3.1415927 -5. -3.1415927 -5. -0. -3.1415927 -5. -3.1415927 -5. -0. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. ], [3.1415927 5. 5. 5. 3.1415927 5. 3.1415927 5. 5. 3.1415927 5. 3.1415927 5. 5. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ], (24,), float32)` |
| Max Episode Steps | `1600`                                                                                                                                                                                                                                                                      |
| Reward Threshold  | `300`                                                                                                                                                                                                                                                                       |
| import            | `gymnasium.make("BipedalWalker-v3")`                                                                                                                                                                                                                                        |

## Description

This is a simple 4-joint walker robot environment.
There are two versions:

- Normal, with slightly uneven terrain.
- Hardcore, with ladders, stumps, pitfalls.

To solve the normal version, you need to get 300 points in 1600 time steps.
To solve the hardcore version, you need 300 points in 2000 time steps.

A heuristic is provided for testing. It’s also useful to get demonstrations
to learn from. To run the heuristic:

```default
python gymnasium/envs/box2d/bipedal_walker.py
```

## Action Space

The action is a `Box(-1, 1, (4,), float32)` with one value per joint, in the order
hip 1, knee 1, hip 2, knee 2. Each value effectively sets the motor torque of its joint:
the sign selects the direction the motor drives the joint, and the absolute value (clipped to `[0, 1]`)
scales the maximum motor torque.

## Observation Space

The observation is a `ndarray` with shape `(24,)`: hull angle, hull angular velocity, horizontal speed,
vertical speed, the angle and angular speed of each of the 4 joints, the ground contact of each leg,
and 10 lidar rangefinder measurements. There are no coordinates in the state vector.

## Rewards

Reward is given for moving forward, totaling 300+ points up to the far end.
If the robot falls (or moves past the left end of the terrain), the reward for that step is -100.
Applying motor torque costs a small amount of points. A more optimal agent will get a better score.

## Starting State

The walker starts standing at the left end of the terrain with the hull
horizontal, and both legs in the same position with a slight knee angle.

## Episode End

The episode will terminate if the hull gets in contact with the ground,
if the walker moves past the left end of the terrain (`x < 0`), or
if the walker exceeds the right end of the terrain length.

The episode is truncated after 1600 steps for `BipedalWalker-v3` and 2000 steps for `BipedalWalkerHardcore-v3`.

## Arguments

To use the *hardcore* environment, you need to specify the `hardcore=True`
(or use `BipedalWalkerHardcore-v3`, which also raises the time limit from 1600 to 2000 steps):

```python
>>> import gymnasium as gym
>>> env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<BipedalWalker<BipedalWalker-v3>>>>>

```

## Version History

- v3: Returns the closest lidar trace instead of furthest;
  faster video recording
- v2: Count energy spent
- v1: Legs now report contact with ground; motors have higher torque and
  speed; ground has higher friction; lidar rendered less nervously.
- v0: Initial version

<!-- ## References -->

## Credits

Created by Oleg Klimov
