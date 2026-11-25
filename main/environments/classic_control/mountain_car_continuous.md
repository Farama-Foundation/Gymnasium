# Mountain Car Continuous

![image](_static/videos/classic_control/mountain_car_continuous.gif)

This environment is part of the [Classic Control environments](../classic_control.md) which contains general information about the environment.

|                   |                                                |
|-------------------|------------------------------------------------|
| Action Space      | `Box(-1.0, 1.0, (1,), float32)`                |
| Observation Space | `Box([-1.2 -0.07], [0.6 0.07], (2,), float32)` |
| Max Episode Steps | `999`                                          |
| Reward Threshold  | `90.0`                                         |
| import            | `gymnasium.make("MountainCarContinuous-v0")`   |

## Description

The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
that can be applied to the car in either direction. The goal of the MDP is to strategically
accelerate the car to reach the goal state on top of the right hill. There are two versions
of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
This version is the one with continuous actions.

This MDP first appeared in [Andrew Moore’s PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

```default
@TECHREPORT{Moore90efficientmemory-based,
    author = {Andrew William Moore},
    title = {Efficient Memory-based Learning for Robot Control},
    institution = {University of Cambridge},
    year = {1990}
}
```

## Observation Space

The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

|   Num | Observation                          |   Min |   Max | Unit         |
|-------|--------------------------------------|-------|-------|--------------|
|     0 | position of the car along the x-axis | -1.2  |  0.6  | position (m) |
|     1 | velocity of the car                  | -0.07 |  0.07 | velocity (v) |

## Action Space

The action is a `ndarray` with shape `(1,)`, representing the directional force applied on the car.
The action is clipped in the range `[-1,1]` and multiplied by a power of 0.0015.

## Transition Dynamics

Given an action, the mountain car follows the following transition dynamics:

*velocity<sub>t+1</sub> = velocity<sub>t</sub> + force \* self.power - 0.0025 \* cos(3 \* position<sub>t</sub>)*

*position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015.
The position is clipped to the range `[-1.2, 0.6]` and velocity is clipped to the range `[-0.07, 0.07]`.
The collision with the left wall (position -1.2) is inelastic, with the velocity set to 0 if the car is moving left.

## Rewards

A negative reward of  *-0.1 \* action<sup>2</sup>* is received at each timestep to penalise for
taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100
is added to the negative reward for that timestep.

## Starting State

The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`.
The starting velocity of the car is always assigned to 0.

## Episode End

The episode ends if either of the following happens:

1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
   and its velocity is greater than or equal to `goal_velocity` (default 0)
2. Truncation: The length of the episode is 999.

## Arguments

Continuous Mountain Car has two parameters for `gymnasium.make` with `render_mode` and `goal_velocity`.
On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

```python
>>> import gymnasium as gym
>>> env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1)  # default goal_velocity=0
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<Continuous_MountainCarEnv<MountainCarContinuous-v0>>>>>
>>> env.reset(seed=123, options={"low": -0.7, "high": -0.5})  # default low=-0.6, high=-0.4
(array([-0.5635296,  0.       ], dtype=float32), {})

```

## Version History

* v0: Initial versions release
