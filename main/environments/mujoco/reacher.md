# Reacher

![image](_static/videos/mujoco/reacher.gif)

This environment is part of the [Mujoco environments](../mujoco.md) which contains general information about the environment.

|                   |                                  |
|-------------------|----------------------------------|
| Action Space      | `Box(-1.0, 1.0, (2,), float32)`  |
| Observation Space | `Box(-inf, inf, (10,), float64)` |
| Max Episode Steps | `50`                             |
| Reward Threshold  | `-3.75`                          |
| import            | `gymnasium.make("Reacher-v5")`   |

## Description

“Reacher” is a two-jointed robot arm.
The goal is to move the robot’s end effector (called *fingertip*) close to a target that is spawned at a random position.

## Action Space

![image](environments/mujoco/action_space_figures/reacher.png)

The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.

|   Num | Action                                                                          |   Control Min |   Control Max | Name (in corresponding XML file)   | Joint   | Type (Unit)   |
|-------|---------------------------------------------------------------------------------|---------------|---------------|------------------------------------|---------|---------------|
|     0 | Torque applied at the first hinge (connecting the link to the point of fixture) |            -1 |             1 | joint0                             | hinge   | torque (N m)  |
|     1 | Torque applied at the second hinge (connecting the two links)                   |            -1 |             1 | joint1                             | hinge   | torque (N m)  |

## Observation Space

The observation space consists of the following parts (in order):

- *cos(qpos) (2 elements):* The cosine of the angles of the two arms.
- *sin(qpos) (2 elements):* The sine of the angles of the two arms.
- *qpos (2 elements):* The coordinates of the target.
- *qvel (2 elements):* The angular velocities of the arms (their derivatives).
- *xpos (2 elements):* The vector between the target and the reacher’s.

The observation space is a `Box(-Inf, Inf, (10,), float64)` where the elements are as follows:

| Num      | Observation                                                                        |   Min |   Max | Name (in corresponding XML file)   | Joint   | Type (Unit)              |
|----------|------------------------------------------------------------------------------------|-------|-------|------------------------------------|---------|--------------------------|
| 0        | cosine of the angle of the first arm                                               |  -inf |   inf | cos(joint0)                        | hinge   | unitless                 |
| 1        | cosine of the angle of the second arm                                              |  -inf |   inf | cos(joint1)                        | hinge   | unitless                 |
| 2        | sine of the angle of the first arm                                                 |  -inf |   inf | sin(joint0)                        | hinge   | unitless                 |
| 3        | sine of the angle of the second arm                                                |  -inf |   inf | sin(joint1)                        | hinge   | unitless                 |
| 4        | x-coordinate of the target                                                         |  -inf |   inf | target_x                           | slide   | position (m)             |
| 5        | y-coordinate of the target                                                         |  -inf |   inf | target_y                           | slide   | position (m)             |
| 6        | angular velocity of the first arm                                                  |  -inf |   inf | joint0                             | hinge   | angular velocity (rad/s) |
| 7        | angular velocity of the second arm                                                 |  -inf |   inf | joint1                             | hinge   | angular velocity (rad/s) |
| 8        | x-value of position_fingertip - position_target                                    |  -inf |   inf | NA                                 | slide   | position (m)             |
| 9        | y-value of position_fingertip - position_target                                    |  -inf |   inf | NA                                 | slide   | position (m)             |
| excluded | z-value of position_fingertip - position_target (constantly 0 since reacher is 2d) |  -inf |   inf | NA                                 | slide   | position (m)             |

Most Gymnasium environments just return the positions and velocities of the joints in the `.xml` file as the state of the environment.
In reacher, however, the state is created by combining only certain elements of the position and velocity and performing some function transformations on them.
The `reacher.xml` contains these 4 joints:

|   Num | Observation                |   Min |   Max | Name (in corresponding XML file)   | Joint   | Unit         |
|-------|----------------------------|-------|-------|------------------------------------|---------|--------------|
|     0 | angle of the first arm     |  -inf |   inf | joint0                             | hinge   | angle (rad)  |
|     1 | angle of the second arm    |  -inf |   inf | joint1                             | hinge   | angle (rad)  |
|     2 | x-coordinate of the target |  -inf |   inf | target_x                           | slide   | position (m) |
|     3 | y-coordinate of the target |  -inf |   inf | target_y                           | slide   | position (m) |

## Rewards

The total reward is:  ***reward***  *=* *reward_dist + reward_ctrl*.

- *reward_dist*:
  This reward is a measure of how far the *fingertip* of the reacher (the unattached end) is from the target,
  with a more negative value assigned if the reacher’s *fingertip* is further away from the target.
  It is $-w_{dist} |(P_{fingertip} - P_{target})|*2$.
  where $w*{dist}$ is the `reward_dist_weight` (default is $1$).
- *reward_ctrl*:
  A negative reward to penalize the reacher for taking actions that are too large.
  It is measured as the negative squared Euclidean norm of the action, i.e. as $-w_{control} |action|*2^2$.
  where $w*{control}$ is the `reward_control_weight` (default is $1$).

`info` contains the individual reward terms (`"reward_dist"` and `"reward_ctrl"`).

## Starting State

The initial position state of the reacher arm is $\\mathcal{U} *{[-0.1 \\times I*{2}, 0.1 \\times I_{2}]}$.
The position state of the goal is (permanently) $\\mathcal{S}(0.2)$.
The initial velocity state of the Reacher arm is $\\mathcal{U} *{[-0.005 \\times 1*{2}, 0.005 \\times 1_{2}]}$.
The velocity state of the object is (permanently) $0_2$.

where $\\mathcal{U}$ is the multivariate uniform continuous distribution and $\\mathcal{S}$ is the uniform continuous spherical distribution.

The default `frame_skip` is $2$, with each MuJoCo timestep lasting $0.01$, so *dt = 2 \* 0.01 = 0.02*.

## Episode End

### Termination

The Reacher never terminates.

### Truncation

The default duration of an episode is 50 timesteps.

## Arguments

Reacher provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
These parameters can be applied during `gymnasium.make` in the following way:

```python
import gymnasium as gym
env = gym.make('Reacher-v5', xml_file=...)
```

| Parameter               | Type      | Default              | Description                                                                                       |
|-------------------------|-----------|----------------------|---------------------------------------------------------------------------------------------------|
| `xml_file`              | **str**   | `"reacher.xml"`      | Path to a MuJoCo model                                                                            |
| `frame_skip`            | **int**   | `2`                  | Number of MuJoCo simulation steps per `step()`, so $dt = frame_skip \\times$ `model.opt.timestep` |
| `default_camera_config` | **dict**  | `{"trackbodyid": 0}` | Properties of the `mj_camera` used for rendering                                                  |
| `reward_dist_weight`    | **float** | `1`                  | Weight for *reward_dist* term (see `Rewards` section)                                             |
| `reward_control_weight` | **float** | `1`                  | Weight for *reward_ctrl* term (see `Rewards` section)                                             |

## Version History

* v5:
  - Minimum `mujoco` version is now 2.3.3.
  - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
  - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
  - Fixed bug: `reward_distance` was based on the state before the physics step, now it is based on the state after the physics step (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/821)).
  - Removed `"z - position_fingertip"` from the observation space since it is always 0 and therefore provides no useful information to the agent, this should result in slightly faster training (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/204)).
  - Added `xml_file` argument.
  - Added `reward_dist_weight`, `reward_control_weight` arguments to configure the reward function (defaults are effectively the same as in `v4`).
  - Fixed `info["reward_ctrl"]`  not being multiplied by the reward weight.
* v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
* v3: This environment does not have a v3 release. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v1: max_time_steps raised to 1000 for robot based tasks (not including reacher, which has a max_time_steps of 50). Added reward_threshold to environments.
* v0: Initial versions release
