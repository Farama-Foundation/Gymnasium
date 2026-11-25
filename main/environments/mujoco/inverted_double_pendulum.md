# Inverted Double Pendulum

![image](_static/videos/mujoco/inverted_double_pendulum.gif)

This environment is part of the [Mujoco environments](../mujoco.md) which contains general information about the environment.

|                   |                                               |
|-------------------|-----------------------------------------------|
| Action Space      | `Box(-1.0, 1.0, (1,), float32)`               |
| Observation Space | `Box(-inf, inf, (9,), float64)`               |
| Max Episode Steps | `1000`                                        |
| Reward Threshold  | `9100.0`                                      |
| import            | `gymnasium.make("InvertedDoublePendulum-v5")` |

## Description

This environment originates from control theory and builds on the cartpole environment based on the work of Barto, Sutton, and Anderson in [“Neuronlike adaptive elements that can solve difficult learning control problems”](https://ieeexplore.ieee.org/document/6313077),
powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity or constraints).
This environment involves a cart that can be moved linearly, with one pole attached to it and a second pole attached to the other end of the first pole (leaving the second pole as the only one with a free end).
The cart can be pushed left or right, and the goal is to balance the second pole on top of the first pole, which is in turn on top of the cart, by applying continuous forces to the cart.

## Action Space

The agent takes a 1-element vector for actions.
The action space is a continuous `(action)` in `[-1, 1]`, where `action` represents the
numerical force applied to the cart (with magnitude representing the amount of force and
sign representing the direction)

|   Num | Action                    |   Control Min |   Control Max | Name (in corresponding XML file)   | Joint   | Type (Unit)   |
|-------|---------------------------|---------------|---------------|------------------------------------|---------|---------------|
|     0 | Force applied on the cart |            -1 |             1 | slider                             | slide   | Force (N)     |

## Observation Space

The observation space consists of the following parts (in order):

- *qpos (1 element):* Position values of the robot’s cart.
- *sin(qpos) (2 elements):* The sine of the angles of poles.
- *cos(qpos) (2 elements):* The cosine of the angles of poles.
- *qvel (3 elements):* The velocities of these individual body parts (their derivatives), clipped to $[-10, 10]$.
- *qfrc_constraint (1 element):* Constraint force of the cart, clipped to $[-10, 10]$.
  There is one constraint force for contacts for each degree of freedom (3).
  The approach and handling of constraints by MuJoCo is unique to the simulator and is based on their research.
  More information can be found  in their [*documentation*](https://mujoco.readthedocs.io/en/latest/computation.html) or in their paper [“Convex and analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo”](https://roboti.us/lab/papers/TodorovICRA14.pdf).

The observation space is a `Box(-Inf, Inf, (9,), float64)` where the elements are as follows:

| Num      | Observation                                                       |   Min |   Max | Name (in corresponding XML file)   | Joint   | Type (Unit)              |
|----------|-------------------------------------------------------------------|-------|-------|------------------------------------|---------|--------------------------|
| 0        | position of the cart along the linear surface                     |  -inf |   inf | slider                             | slide   | position (m)             |
| 1        | sine of the angle between the cart and the first pole             |  -inf |   inf | sin(hinge)                         | hinge   | unitless                 |
| 2        | sine of the angle between the two poles                           |  -inf |   inf | sin(hinge2)                        | hinge   | unitless                 |
| 3        | cosine of the angle between the cart and the first pole           |  -inf |   inf | cos(hinge)                         | hinge   | unitless                 |
| 4        | cosine of the angle between the two poles                         |  -inf |   inf | cos(hinge2)                        | hinge   | unitless                 |
| 5        | velocity of the cart                                              |  -inf |   inf | slider                             | slide   | velocity (m/s)           |
| 6        | angular velocity of the angle between the cart and the first pole |  -inf |   inf | hinge                              | hinge   | angular velocity (rad/s) |
| 7        | angular velocity of the angle between the two poles               |  -inf |   inf | hinge2                             | hinge   | angular velocity (rad/s) |
| 8        | constraint force on the cart                                      |  -inf |   inf | slider                             | slide   | Force (N)                |
| excluded | constraint torque on the first hinge                              |  -inf |   inf | hinge                              | hinge   | Torque (N m)             |
| excluded | constraint torque on the second hinge                             |  -inf |   inf | hinge2                             | hinge   | Torque (N m)             |

## Rewards

The total reward is:  ***reward***  *=* *alive_bonus - distance_penalty - velocity_penalty*.

- *alive_bonus*:
  Every timestep that the Inverted Pendulum is healthy (see definition in section “Episode End”),
  it gets a reward of fixed value `healthy_reward` (default is $10$).
- *distance_penalty*:
  This reward is a measure of how far the *tip* of the second pendulum (the only free end) moves,
  and it is calculated as $0.01 x_{pole2-tip}^2 + (y_{pole2-tip}-2)^2$,
  where $x_{pole2-tip}, y_{pole2-tip}$ are the xy-coordinates of the tip of the second pole.
- *velocity_penalty*:
  A negative reward to penalize the agent for moving too fast.
  $10^{-3} \\omega_1^2 + 5 \\times 10^{-3} \\omega_2^2$,
  where $\\omega_1, \\omega_2$ are the angular velocities of the hinges.

`info` contains the individual reward terms.

## Starting State

The initial position state is $\\mathcal{U} *{[-reset_noise_scale \\times I*{3}, reset_noise_scale \\times I_{3}]}$.
The initial velocity state is $\\mathcal{N}(0_{3}, reset_noise_scale^2 \\times I_{3})$.

where $\\mathcal{N}$ is the multivariate normal distribution and $\\mathcal{U}$ is the multivariate uniform continuous distribution.

## Episode End

### Termination

The environment terminates when the Inverted Double Pendulum is unhealthy.
The Inverted Double Pendulum is unhealthy if any of the following happens:

1. The y-coordinate (height) of the tip of the second pole $\\leq 1$.

Note: The maximum standing height of the system is 1.2 m when all the parts are perpendicularly vertical on top of each other.

### Truncation

The default duration of an episode is 1000 timesteps.

## Arguments

InvertedDoublePendulum provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
These parameters can be applied during `gymnasium.make` in the following way:

```python
import gymnasium as gym
env = gym.make('InvertedDoublePendulum-v5', healthy_reward=10, ...)
```

| Parameter               | Type      | Default                          | Description                                                                                       |
|-------------------------|-----------|----------------------------------|---------------------------------------------------------------------------------------------------|
| `xml_file`              | **str**   | `"inverted_double_pendulum.xml"` | Path to a MuJoCo model                                                                            |
| `frame_skip`            | **int**   | `5`                              | Number of MuJoCo simulation steps per `step()`, so $dt = frame_skip \\times$ `model.opt.timestep` |
| `default_camera_config` | **dict**  | `None`                           | Properties of the `mj_camera` used for rendering                                                  |
| `healthy_reward`        | **float** | `10`                             | Constant reward given if the pendulum is `healthy` (upright) (see `Rewards` section)              |
| `reset_noise_scale`     | **float** | `0.1`                            | Scale of random perturbations of initial position and velocity (see `Starting State` section)     |

## Version History

* v5:
  - Minimum `mujoco` version is now 2.3.3.
  - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
  - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
  - Fixed bug: `healthy_reward` was given on every step (even if the Pendulum is unhealthy), now it is only given if the DoublePendulum is healthy (not terminated)(related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/500)).
  - Excluded the `qfrc_constraint` (“constraint force”) of the hinges from the observation space (as it was always 0, thus providing no useful information to the agent, resulting in slightly faster training) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/228)).
  - Added `xml_file` argument.
  - Added `reset_noise_scale` argument to set the range of initial states.
  - Added `healthy_reward` argument to configure the reward function (defaults are effectively the same as in `v4`).
  - Added individual reward terms in `info` (`info["reward_survive"]`, `info["distance_penalty"]`, `info["velocity_penalty"]`).
* v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
* v3: This environment does not have a v3 release. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum).
* v0: Initial versions release.
