# Mountain Car

![image](_static/videos/classic_control/mountain_car.gif)

This environment is part of the [Classic Control environments](../classic_control.md) which contains general information about the environment.

|                   |                                                |
|-------------------|------------------------------------------------|
| Action Space      | `Discrete(3)`                                  |
| Observation Space | `Box([-1.2 -0.07], [0.6 0.07], (2,), float32)` |
| Max Episode Steps | `200`                                          |
| Reward Threshold  | `-110.0`                                       |
| import            | `gymnasium.make("MountainCar-v0")`             |

## Description

The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
that can be applied to the car in either direction. The goal of the MDP is to strategically
accelerate the car to reach the goal state on top of the right hill. There are two versions
of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
This version is the one with discrete actions.

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

There are 3 discrete deterministic actions:

- 0: Accelerate to the left
- 1: Don’t accelerate
- 2: Accelerate to the right

## Transition Dynamics

Given an action, the mountain car follows the following transition dynamics:

*velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) \* force - cos(3 \* position<sub>t</sub>) \* gravity*

*position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

where force = 0.001 and gravity = 0.0025. The position is clipped to the range `[-1.2, 0.6]` and
velocity is clipped to the range `[-0.07, 0.07]`. The collision with the left wall (position -1.2) is inelastic,
with the velocity set to 0 if the car is moving left.

## Rewards

The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
penalised with a reward of -1 for each timestep.

## Starting State

The position of the car is assigned a uniform random value in  *[-0.6 , -0.4]*.
The starting velocity of the car is always assigned to 0.

## Episode End

The episode ends if either of the following happens:

1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
   and its velocity is greater than or equal to `goal_velocity` (default 0)
2. Truncation: The length of the episode is 200.

## Arguments

Mountain Car has two parameters for `gymnasium.make` with `render_mode` and `goal_velocity`.
On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

```python
>>> import gymnasium as gym
>>> env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.05)  # default goal_velocity=0
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<MountainCarEnv<MountainCar-v0>>>>>
>>> env.reset(seed=123, options={"low": -0.7, "high": 0.5})  # default low=-0.6, high=-0.4
(array([0.11882224, 0.        ], dtype=float32), {})

```

## Version History

* v0: Initial versions release
