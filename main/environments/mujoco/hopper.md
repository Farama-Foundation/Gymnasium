# Hopper

![image](_static/videos/mujoco/hopper.gif)

This environment is part of the [Mujoco environments](../mujoco.md) which contains general information about the environment.

|                   |                                  |
|-------------------|----------------------------------|
| Action Space      | `Box(-1.0, 1.0, (3,), float32)`  |
| Observation Space | `Box(-inf, inf, (11,), float64)` |
| Max Episode Steps | `1000`                           |
| Reward Threshold  | `3800.0`                         |
| import            | `gymnasium.make("Hopper-v5")`    |

## Description

This environment is based on the work of Erez, Tassa, and Todorov in [“Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks”](https://www.roboticsproceedings.org/rss07/p10.pdf).
The environment aims to increase the number of independent state and control variables compared to classical control environments.
The hopper is a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests.
The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts.

## Action Space

![image](environments/mujoco/action_space_figures/hopper.png)

The action space is a `Box(-1, 1, (3,), float32)`. An action represents the torques applied at the hinge joints.

|   Num | Action                            |   Control Min |   Control Max | Name (in corresponding XML file)   | Joint   | Type (Unit)   |
|-------|-----------------------------------|---------------|---------------|------------------------------------|---------|---------------|
|     0 | Torque applied on the thigh rotor |            -1 |             1 | thigh_joint                        | hinge   | torque (N m)  |
|     1 | Torque applied on the leg rotor   |            -1 |             1 | leg_joint                          | hinge   | torque (N m)  |
|     2 | Torque applied on the foot rotor  |            -1 |             1 | foot_joint                         | hinge   | torque (N m)  |

## Observation Space

The observation space consists of the following parts (in order):

- *qpos (5 elements by default):* Position values of the robot’s body parts.
- *qvel (6 elements):* The velocities of these individual body parts (their derivatives), clipped to $[-10, 10]$.

By default, the observation does not include the robot’s x-coordinate (`rootx`).
This can  be included by passing `exclude_current_positions_from_observation=False` during construction.
In this case, the observation space will be a `Box(-Inf, Inf, (12,), float64)`, where the first observation element is the x-coordinate of the robot.
Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x-coordinate is returned in `info` with the key `"x_position"`.

By default, however, the observation space is a `Box(-Inf, Inf, (11,), float64)` where the elements are as follows:

| Num      | Observation                                        |   Min |   Max | Name (in corresponding XML file)   | Joint   | Type (Unit)              |
|----------|----------------------------------------------------|-------|-------|------------------------------------|---------|--------------------------|
| 0        | z-coordinate of the torso (height of hopper)       |  -inf |   inf | rootz                              | slide   | position (m)             |
| 1        | angle of the torso                                 |  -inf |   inf | rooty                              | hinge   | angle (rad)              |
| 2        | angle of the thigh joint                           |  -inf |   inf | thigh_joint                        | hinge   | angle (rad)              |
| 3        | angle of the leg joint                             |  -inf |   inf | leg_joint                          | hinge   | angle (rad)              |
| 4        | angle of the foot joint                            |  -inf |   inf | foot_joint                         | hinge   | angle (rad)              |
| 5        | velocity of the x-coordinate of the torso          |  -inf |   inf | rootx                              | slide   | velocity (m/s)           |
| 6        | velocity of the z-coordinate (height) of the torso |  -inf |   inf | rootz                              | slide   | velocity (m/s)           |
| 7        | angular velocity of the angle of the torso         |  -inf |   inf | rooty                              | hinge   | angular velocity (rad/s) |
| 8        | angular velocity of the thigh hinge                |  -inf |   inf | thigh_joint                        | hinge   | angular velocity (rad/s) |
| 9        | angular velocity of the leg hinge                  |  -inf |   inf | leg_joint                          | hinge   | angular velocity (rad/s) |
| 10       | angular velocity of the foot hinge                 |  -inf |   inf | foot_joint                         | hinge   | angular velocity (rad/s) |
| excluded | x-coordinate of the torso                          |  -inf |   inf | rootx                              | slide   | position (m)             |

## Rewards

The total reward is:  ***reward***  *=* *healthy_reward + forward_reward - ctrl_cost*.

- *healthy_reward*:
  Every timestep that the Hopper is healthy (see definition in section “Episode End”),
  it gets a reward of fixed value `healthy_reward` (default is $1$).
- *forward_reward*:
  A reward for moving forward,
  this reward would be positive if the Hopper moves forward (in the positive $x$ direction / in the right direction).
  $w_{forward} \\times \\frac{dx}{dt}$, where
  $dx$ is the displacement of the “torso” ($x_{after-action} - x_{before-action}$),
  $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $4$),
  and `frametime` which is $0.002$ - so the default is $dt = 4 \\times 0.002 = 0.008$,
  $w_{forward}$ is the `forward_reward_weight` (default is $1$).
- *ctrl_cost*:
  A negative reward to penalize the Hopper for taking actions that are too large.
  $w_{control} \\times |action|*2^2$,
  where $w*{control}$ is `ctrl_cost_weight` (default is $10^{-3}$).

`info` contains the individual reward terms.

## Starting State

The initial position state is $[0, 1.25, 0, 0, 0, 0] + \\mathcal{U} *{[-reset_noise_scale \\times I*{6}, reset_noise_scale \\times I_{6}]}$.
The initial velocity state is $\\mathcal{U} *{[-reset_noise_scale \\times I*{6}, reset_noise_scale \\times I_{6}]}$.

where $\\mathcal{U}$ is the multivariate uniform continuous distribution.

Note that the z-coordinate is non-zero so that the hopper can stand up immediately.

## Episode End

### Termination

If `terminate_when_unhealthy is True` (the default), the environment terminates when the Hopper is unhealthy.
The Hopper is unhealthy if any of the following happens:

1. An element of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, otherwise `observation[2:]`), with the velocities taken before clipping, is no longer contained in the open interval specified by the `healthy_state_range` argument (default is $(-100, 100)$).
2. The height of the hopper (`observation[0]` if  `exclude_current_positions_from_observation=True`, otherwise `observation[1]`) is no longer contained in the open interval specified by the `healthy_z_range` argument (default is $(0.7, +\\infty)$) (usually meaning that it has fallen).
3. The angle of the torso (`observation[1]` if  `exclude_current_positions_from_observation=True`, otherwise `observation[2]`) is no longer contained in the open interval specified by the `healthy_angle_range` argument (default is $(-0.2, 0.2)$).

### Truncation

The default duration of an episode is 1000 timesteps.

## Arguments

Hopper provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
These parameters can be applied during `gymnasium.make` in the following way:

```python
import gymnasium as gym
env = gym.make('Hopper-v5', ctrl_cost_weight=1e-3, ....)
```

| Parameter                                    | Type      | Default                                                                                         | Description                                                                                                                                                                                                 |
|----------------------------------------------|-----------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `xml_file`                                   | **str**   | `"hopper.xml"`                                                                                  | Path to a MuJoCo model                                                                                                                                                                                      |
| `frame_skip`                                 | **int**   | `4`                                                                                             | Number of MuJoCo simulation steps per `step()`, so $dt = frame_skip \\times$ `model.opt.timestep`                                                                                                           |
| `default_camera_config`                      | **dict**  | `{"trackbodyid": 2, "distance": 3.0, "lookat": np.array((0.0, 0.0, 1.15)), "elevation": -20.0}` | Properties of the `mj_camera` used for rendering                                                                                                                                                            |
| `forward_reward_weight`                      | **float** | `1`                                                                                             | Weight for *forward_reward* term (see `Rewards` section)                                                                                                                                                    |
| `ctrl_cost_weight`                           | **float** | `1e-3`                                                                                          | Weight for *ctrl_cost* reward (see `Rewards` section)                                                                                                                                                       |
| `healthy_reward`                             | **float** | `1`                                                                                             | Weight for *healthy_reward* reward (see `Rewards` section)                                                                                                                                                  |
| `terminate_when_unhealthy`                   | **bool**  | `True`                                                                                          | If `True`, issue a `terminated` signal is unhealthy (see `Episode End` section)                                                                                                                             |
| `healthy_state_range`                        | **tuple** | `(-100, 100)`                                                                                   | The elements of `observation[1:]` (if `exclude_current_positions_from_observation=True`, else `observation[2:]`) must be in this range for the hopper to be considered healthy (see `Episode End` section)  |
| `healthy_z_range`                            | **tuple** | `(0.7, float("inf"))`                                                                           | The z-coordinate must be in this range for the hopper to be considered healthy (see `Episode End` section)                                                                                                  |
| `healthy_angle_range`                        | **tuple** | `(-0.2, 0.2)`                                                                                   | The angle given by `observation[1]` (if `exclude_current_positions_from_observation=True`, else `observation[2]`) must be in this range for the hopper to be considered healthy (see `Episode End` section) |
| `reset_noise_scale`                          | **float** | `5e-3`                                                                                          | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
| `exclude_current_positions_from_observation` | **bool**  | `True`                                                                                          | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies(see `Observation Space` section)          |

## Version History

* v5:
  - Minimum `mujoco` version is now 2.3.3.
  - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
  - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
  - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
  - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
  - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
  - Fixed bug: `healthy_reward` was given on every step (even if the Hopper was unhealthy), now it is only given when the Hopper is healthy. The `info["reward_survive"]` is updated with this change (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/526)).
  - Restored the `xml_file` argument (was removed in `v4`).
  - Added individual reward terms in `info` (`info["reward_forward"]`, `info["reward_ctrl"]`, `info["reward_survive"]`).
  - Added `info["z_distance_from_origin"]` which is equal to the vertical distance of the “torso” body from its initial position.
* v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
* v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen). Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
* v0: Initial versions release.
