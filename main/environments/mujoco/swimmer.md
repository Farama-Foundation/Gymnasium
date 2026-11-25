# Swimmer

![image](_static/videos/mujoco/swimmer.gif)

This environment is part of the [Mujoco environments](../mujoco.md) which contains general information about the environment.

|                   |                                 |
|-------------------|---------------------------------|
| Action Space      | `Box(-1.0, 1.0, (2,), float32)` |
| Observation Space | `Box(-inf, inf, (8,), float64)` |
| Max Episode Steps | `1000`                          |
| Reward Threshold  | `360.0`                         |
| import            | `gymnasium.make("Swimmer-v5")`  |

## Description

This environment corresponds to the Swimmer environment described in Rémi Coulom’s PhD thesis [“Reinforcement Learning Using Neural Networks, with Applications to Motor Control”](https://tel.archives-ouvertes.fr/tel-00003985/document).
The environment aims to increase the number of independent state and control variables compared to classical control environments.
The swimmers consist of three or more segments (’ ***links***’) and one less articulation joints (’ ***rotors***’) - one rotor joint connects exactly two links to form a linear chain.
The swimmer is suspended in a two-dimensional pool and always starts in the same position (subject to some deviation drawn from a uniform distribution),
and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction.

## Notes

The problem parameters are:
Problem parameters:

* *n*: number of body parts
* *m<sub>i</sub>*: mass of part *i* (*i* ∈ {1…n})
* *l<sub>i</sub>*: length of part *i* (*i* ∈ {1…n})
* *k*: viscous-friction coefficient

While the default environment has *n* = 3, *l<sub>i</sub>* = 1 (with a capsule radius of 0.1), and *k* = 0.1.
It is possible to pass a custom MuJoCo XML file during construction to increase the number of links, or to tweak any of the parameters.

## Action Space

![image](environments/mujoco/action_space_figures/swimmer.png)

The action space is a `Box(-1, 1, (2,), float32)`. An action represents the torques applied between *links*

|   Num | Action                             |   Control Min |   Control Max | Name (in corresponding XML file)   | Joint   | Type (Unit)   |
|-------|------------------------------------|---------------|---------------|------------------------------------|---------|---------------|
|     0 | Torque applied on the first rotor  |            -1 |             1 | motor1_rot                         | hinge   | torque (N m)  |
|     1 | Torque applied on the second rotor |            -1 |             1 | motor2_rot                         | hinge   | torque (N m)  |

## Observation Space

The observation space consists of the following parts (in order):

- *qpos (3 elements by default):* Position values of the robot’s body parts.
- *qvel (5 elements):* The velocities of these individual body parts (their derivatives).

By default, the observation does not include the x- and y-coordinates of the front tip.
These can be included by passing `exclude_current_positions_from_observation=False` during construction.
In this case, the observation space will be a `Box(-Inf, Inf, (10,), float64)`, where the first two observations are the x- and y-coordinates of the front tip.
Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

By default, however, the observation space is a `Box(-Inf, Inf, (8,), float64)` where the elements are as follows:

| Num      | Observation                          |   Min |   Max | Name (in corresponding XML file)   | Joint   | Type (Unit)              |
|----------|--------------------------------------|-------|-------|------------------------------------|---------|--------------------------|
| 0        | angle of the front tip               |  -inf |   inf | free_body_rot                      | hinge   | angle (rad)              |
| 1        | angle of the first rotor             |  -inf |   inf | motor1_rot                         | hinge   | angle (rad)              |
| 2        | angle of the second rotor            |  -inf |   inf | motor2_rot                         | hinge   | angle (rad)              |
| 3        | velocity of the tip along the x-axis |  -inf |   inf | slider1                            | slide   | velocity (m/s)           |
| 4        | velocity of the tip along the y-axis |  -inf |   inf | slider2                            | slide   | velocity (m/s)           |
| 5        | angular velocity of front tip        |  -inf |   inf | free_body_rot                      | hinge   | angular velocity (rad/s) |
| 6        | angular velocity of first rotor      |  -inf |   inf | motor1_rot                         | hinge   | angular velocity (rad/s) |
| 7        | angular velocity of second rotor     |  -inf |   inf | motor2_rot                         | hinge   | angular velocity (rad/s) |
| excluded | position of the tip along the x-axis |  -inf |   inf | slider1                            | slide   | position (m)             |
| excluded | position of the tip along the y-axis |  -inf |   inf | slider2                            | slide   | position (m)             |

## Rewards

The total reward is:  ***reward***  *=* *forward_reward - ctrl_cost*.

- *forward_reward*:
  A reward for moving forward,
  this reward would be positive if the Swimmer moves forward (in the positive $x$ direction / in the right direction).
  $w_{forward} \\times \\frac{dx}{dt}$, where
  $dx$ is the displacement of the (front) “tip” ($x_{after-action} - x_{before-action}$),
  $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is 4),
  and `frametime` which is $0.01$ - so the default is $dt = 4 \\times 0.01 = 0.04$,
  $w_{forward}$ is the `forward_reward_weight` (default is $1$).
- *ctrl_cost*:
  A negative reward to penalize the Swimmer for taking actions that are too large.
  $w_{control} \\times |action|*2^2$,
  where $w*{control}$ is `ctrl_cost_weight` (default is $10^{-4}$).

`info` contains the individual reward terms.

## Starting State

The initial position state is $\\mathcal{U} *{[-reset_noise_scale \\times I*{5}, reset_noise_scale \\times I_{5}]}$.
The initial velocity state is $\\mathcal{U} *{[-reset_noise_scale \\times I*{5}, reset_noise_scale \\times I_{5}]}$.

where $\\mathcal{U}$ is the multivariate uniform continuous distribution.

## Episode End

### Termination

The Swimmer never terminates.

### Truncation

The default duration of an episode is 1000 timesteps.

## Arguments

Swimmer provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
These parameters can be applied during `gymnasium.make` in the following way:

```python
import gymnasium as gym
env = gym.make('Swimmer-v5', xml_file=...)
```

| Parameter                                    | Type      | Default         | Description                                                                                                                                                                                                 |
|----------------------------------------------|-----------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `xml_file`                                   | **str**   | `"swimmer.xml"` | Path to a MuJoCo model                                                                                                                                                                                      |
| `frame_skip`                                 | **int**   | `4`             | Number of MuJoCo simulation steps per `step()`, so $dt = frame_skip \\times$ `model.opt.timestep`                                                                                                           |
| `default_camera_config`                      | **dict**  | `None`          | Properties of the `mj_camera` used for rendering                                                                                                                                                            |
| `forward_reward_weight`                      | **float** | `1`             | Weight for *forward_reward* term (see `Rewards` section)                                                                                                                                                    |
| `ctrl_cost_weight`                           | **float** | `1e-4`          | Weight for *ctrl_cost* term (see `Rewards` section)                                                                                                                                                         |
| `reset_noise_scale`                          | **float** | `0.1`           | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
| `exclude_current_positions_from_observation` | **bool**  | `True`          | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation Space` section) |

## Version History

* v5:
  - Minimum `mujoco` version is now 2.3.3.
  - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
  - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
  - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
  - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
  - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
  - Restored the `xml_file` argument (was removed in `v4`).
  - Added `forward_reward_weight`, `ctrl_cost_weight`, to configure the reward function (defaults are effectively the same as in `v4`).
  - Added `reset_noise_scale` argument to set the range of initial states.
  - Added `exclude_current_positions_from_observation` argument.
  - Replaced `info["reward_fwd"]` and `info["forward_reward"]` with `info["reward_forward"]` to be consistent with the other environments.
* v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
* v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen). Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
* v0: Initial versions release.
