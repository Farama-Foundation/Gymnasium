# Pendulum

![image](_static/videos/classic_control/pendulum.gif)

This environment is part of the [Classic Control environments](../classic_control.md) which contains general information about the environment.

|                   |                                                 |
|-------------------|-------------------------------------------------|
| Action Space      | `Box(-2.0, 2.0, (1,), float32)`                 |
| Observation Space | `Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)` |
| Max Episode Steps | `200`                                           |
| import            | `gymnasium.make("Pendulum-v1")`                 |

## Description

The inverted pendulum swingup problem is based on the classic problem in control theory.
The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
into an upright position, with its center of gravity right above the fixed point.

The diagram below specifies the coordinate system used for the implementation of the pendulum’s
dynamic equations.

![Pendulum Coordinate System](_static/diagrams/pendulum.png)

- `x-y`: cartesian coordinates of the pendulum’s end in meters.
- `theta` : angle in radians.
- `tau`: torque in `N m`. Defined as positive *counter-clockwise*.

## Action Space

The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

|   Num | Action   |   Min |   Max |
|-------|----------|-------|-------|
|     0 | Torque   |    -2 |     2 |

## Observation Space

The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum’s free
end and its angular velocity.

|   Num | Observation      |   Min |   Max |
|-------|------------------|-------|-------|
|     0 | x = cos(theta)   |    -1 |     1 |
|     1 | y = sin(theta)   |    -1 |     1 |
|     2 | Angular Velocity |    -8 |     8 |

## Rewards

The reward function is defined as:

*r = -(theta<sup>2</sup> + 0.1 \* theta_dt<sup>2</sup> + 0.001 \* torque<sup>2</sup>)*

where `theta` is the pendulum’s angle normalized between  *[-pi, pi]* (with 0 being in the upright position).
Based on the above equation, the minimum reward that can be obtained is
 *-(pi<sup>2</sup> + 0.1 \* 8<sup>2</sup> + 0.001 \* 2<sup>2</sup>) = -16.2736044*,
while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

## Starting State

The starting state is a random angle in  *[-pi, pi]* and a random angular velocity in  *[-1,1]*.

## Episode End

The episode is truncated at 200 time steps. There is no termination condition.

## Arguments

Pendulum has two parameters for `gymnasium.make` with `render_mode` and `g` representing
the acceleration of gravity measured in  *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
The default value is `g = 10.0`.
On reset, the `options` parameter allows the user to change the bounds used to determine the new random state:
`x_init` bounds the initial angle `theta` and `y_init` bounds the initial angular velocity, which are sampled
uniformly from `[-x_init, x_init]` and `[-y_init, y_init]` respectively.

```python
>>> import gymnasium as gym
>>> env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)  # default g=10.0
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<PendulumEnv<Pendulum-v1>>>>>
>>> env.reset(seed=123, options={"x_init": np.pi/2, "y_init": 0.5})  # default x_init=np.pi, y_init=1.0
(array([ 0.8403459 ,  0.54205054, -0.44617897], dtype=float32), {})

```

## Version History

* v1: Simplify the math equations, no difference in behavior.
* v0: Initial versions release
