# Half Cheetah

![image](_static/videos/mujoco/half_cheetah.gif)

This environment is part of the [Mujoco environments](../mujoco.md) which contains general information about the environment.

|                   |                                    |
|-------------------|------------------------------------|
| Action Space      | `Box(-1.0, 1.0, (6,), float32)`    |
| Observation Space | `Box(-inf, inf, (17,), float64)`   |
| Max Episode Steps | `1000`                             |
| Reward Threshold  | `4800.0`                           |
| import            | `gymnasium.make("HalfCheetah-v5")` |

## Description

This environment is based on the work of P. Wawrzyński in [“A Cat-Like Robot Real-Time Learning to Run”](http://staff.elka.pw.edu.pl/~pwawrzyn/pub-s/0812_LSCLRR.pdf).
The HalfCheetah is a 2-dimensional robot consisting of 7 body parts (a torso with a head, and a thigh, shin and foot on each of the two legs) connected by 6 hinge joints.
The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward.
The cheetah’s torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins).

## Action Space

![image](environments/mujoco/action_space_figures/half_cheetah.png)

The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.

|   Num | Action                                  |   Control Min |   Control Max | Name (in corresponding XML file)   | Joint   | Type (Unit)   |
|-------|-----------------------------------------|---------------|---------------|------------------------------------|---------|---------------|
|     0 | Torque applied on the back thigh rotor  |            -1 |             1 | bthigh                             | hinge   | torque (N m)  |
|     1 | Torque applied on the back shin rotor   |            -1 |             1 | bshin                              | hinge   | torque (N m)  |
|     2 | Torque applied on the back foot rotor   |            -1 |             1 | bfoot                              | hinge   | torque (N m)  |
|     3 | Torque applied on the front thigh rotor |            -1 |             1 | fthigh                             | hinge   | torque (N m)  |
|     4 | Torque applied on the front shin rotor  |            -1 |             1 | fshin                              | hinge   | torque (N m)  |
|     5 | Torque applied on the front foot rotor  |            -1 |             1 | ffoot                              | hinge   | torque (N m)  |

## Observation Space

The observation space consists of the following parts (in order):

- *qpos (8 elements by default):* Position values of the robot’s body parts.
- *qvel (9 elements):* The velocities of these individual body parts (their derivatives).

By default, the observation does not include the robot’s x-coordinate (`rootx`).
This can be included by passing `exclude_current_positions_from_observation=False` during construction.
In this case, the observation space will be a `Box(-Inf, Inf, (18,), float64)`, where the first observation element is the x-coordinate of the robot.
Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x-coordinate is returned in `info` with the key `"x_position"`.

By default, however, the observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:

| Num      | Observation                               |   Min |   Max | Name (in corresponding XML file)   | Joint   | Type (Unit)              |
|----------|-------------------------------------------|-------|-------|------------------------------------|---------|--------------------------|
| 0        | z-coordinate of the front tip             |  -inf |   inf | rootz                              | slide   | position (m)             |
| 1        | angle of the front tip                    |  -inf |   inf | rooty                              | hinge   | angle (rad)              |
| 2        | angle of the back thigh                   |  -inf |   inf | bthigh                             | hinge   | angle (rad)              |
| 3        | angle of the back shin                    |  -inf |   inf | bshin                              | hinge   | angle (rad)              |
| 4        | angle of the back foot                    |  -inf |   inf | bfoot                              | hinge   | angle (rad)              |
| 5        | angle of the front thigh                  |  -inf |   inf | fthigh                             | hinge   | angle (rad)              |
| 6        | angle of the front shin                   |  -inf |   inf | fshin                              | hinge   | angle (rad)              |
| 7        | angle of the front foot                   |  -inf |   inf | ffoot                              | hinge   | angle (rad)              |
| 8        | velocity of the x-coordinate of front tip |  -inf |   inf | rootx                              | slide   | velocity (m/s)           |
| 9        | velocity of the z-coordinate of front tip |  -inf |   inf | rootz                              | slide   | velocity (m/s)           |
| 10       | angular velocity of the front tip         |  -inf |   inf | rooty                              | hinge   | angular velocity (rad/s) |
| 11       | angular velocity of the back thigh        |  -inf |   inf | bthigh                             | hinge   | angular velocity (rad/s) |
| 12       | angular velocity of the back shin         |  -inf |   inf | bshin                              | hinge   | angular velocity (rad/s) |
| 13       | angular velocity of the back foot         |  -inf |   inf | bfoot                              | hinge   | angular velocity (rad/s) |
| 14       | angular velocity of the front thigh       |  -inf |   inf | fthigh                             | hinge   | angular velocity (rad/s) |
| 15       | angular velocity of the front shin        |  -inf |   inf | fshin                              | hinge   | angular velocity (rad/s) |
| 16       | angular velocity of the front foot        |  -inf |   inf | ffoot                              | hinge   | angular velocity (rad/s) |
| excluded | x-coordinate of the front tip             |  -inf |   inf | rootx                              | slide   | position (m)             |

## Rewards

The total reward is:  ***reward***  *=* *forward_reward - ctrl_cost*.

- *forward_reward*:
  A reward for moving forward,
  this reward would be positive if the Half Cheetah moves forward (in the positive $x$ direction / in the right direction).
  $w_{forward} \\times \\frac{dx}{dt}$, where
  $dx$ is the displacement of the “tip” ($x_{after-action} - x_{before-action}$),
  $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
  and `frametime` which is $0.01$ - so the default is $dt = 5 \\times 0.01 = 0.05$,
  $w_{forward}$ is the `forward_reward_weight` (default is $1$).
- *ctrl_cost*:
  A negative reward to penalize the Half Cheetah for taking actions that are too large.
  $w_{control} \\times |action|*2^2$,
  where $w*{control}$ is `ctrl_cost_weight` (default is $0.1$).

`info` contains the individual reward terms.

## Starting State

The initial position state is $\\mathcal{U} *{[-reset_noise_scale \\times I*{9}, reset_noise_scale \\times I_{9}]}$.
The initial velocity state is $\\mathcal{N}(0_{9}, reset_noise_scale^2 \\times I_{9})$.

where $\\mathcal{N}$ is the multivariate normal distribution and $\\mathcal{U}$ is the multivariate uniform continuous distribution.

## Episode End

### Termination

The Half Cheetah never terminates.

### Truncation

The default duration of an episode is 1000 timesteps.

## Arguments

HalfCheetah provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
These parameters can be applied during `gymnasium.make` in the following way:

```python
import gymnasium as gym
env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1, ....)
```

| Parameter                                    | Type      | Default              | Description                                                                                                                                                                                         |
|----------------------------------------------|-----------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `xml_file`                                   | **str**   | `"half_cheetah.xml"` | Path to a MuJoCo model                                                                                                                                                                              |
| `frame_skip`                                 | **int**   | `5`                  | Number of MuJoCo simulation steps per `step()`, so $dt = frame_skip \\times$ `model.opt.timestep`                                                                                                   |
| `default_camera_config`                      | **dict**  | `{"distance": 4.0}`  | Properties of the `mj_camera` used for rendering                                                                                                                                                    |
| `forward_reward_weight`                      | **float** | `1`                  | Weight for *forward_reward* term (see `Rewards` section)                                                                                                                                            |
| `ctrl_cost_weight`                           | **float** | `0.1`                | Weight for *ctrl_cost* weight (see `Rewards` section)                                                                                                                                               |
| `reset_noise_scale`                          | **float** | `0.1`                | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                       |
| `exclude_current_positions_from_observation` | **bool**  | `True`               | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation Space` section) |

## Version History

* v5:
  - Minimum `mujoco` version is now 2.3.3.
  - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
  - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
  - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
  - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
  - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
  - Restored the `xml_file` argument (was removed in `v4`).
  - Renamed `info["reward_run"]` to `info["reward_forward"]` to be consistent with the other environments.
  - Replaced the deprecated `settotalmass="14"` compiler attribute in `half_cheetah.xml` with an equivalent geom `density`, the total mass remains 14 (backported to `v4`).
* v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
* v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen). Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
* v0: Initial versions release.
