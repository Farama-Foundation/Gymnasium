# Inverted Pendulum

![image](_static/videos/mujoco/inverted_pendulum.gif)

This environment is part of the [Mujoco environments](../mujoco.md) which contains general information about the environment.

|                   |                                         |
|-------------------|-----------------------------------------|
| Action Space      | `Box(-3.0, 3.0, (1,), float32)`         |
| Observation Space | `Box(-inf, inf, (4,), float64)`         |
| Max Episode Steps | `1000`                                  |
| Reward Threshold  | `950.0`                                 |
| import            | `gymnasium.make("InvertedPendulum-v5")` |

## Description

This environment is the Cartpole environment, based on the work of Barto, Sutton, and Anderson in [“Neuronlike adaptive elements that can solve difficult learning control problems”](https://ieeexplore.ieee.org/document/6313077),
just like in the classic environments, but now powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity).
This environment consists of a cart that can be moved linearly, with a pole attached to one end and having another end free.
The cart can be pushed left or right, and the goal is to balance the pole on top of the cart by applying forces to the cart.

## Action Space

The agent takes a 1-element vector for actions.

The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
the numerical force applied to the cart (with magnitude representing the amount of
force and sign representing the direction)

|   Num | Action                    |   Control Min |   Control Max | Name (in corresponding XML file)   | Joint   | Type (Unit)   |
|-------|---------------------------|---------------|---------------|------------------------------------|---------|---------------|
|     0 | Force applied on the cart |            -3 |             3 | slider                             | slide   | Force (N)     |

## Observation Space

The observation space consists of the following parts (in order):

- *qpos (2 elements):* Position values of the robot’s cart and pole.
- *qvel (2 elements):* The velocities of cart and pole (their derivatives).

The observation space is a `Box(-Inf, Inf, (4,), float64)` where the elements are as follows:

|   Num | Observation                                   |   Min |   Max | Name (in corresponding XML file)   | Joint   | Type (Unit)              |
|-------|-----------------------------------------------|-------|-------|------------------------------------|---------|--------------------------|
|     0 | position of the cart along the linear surface |  -inf |   inf | slider                             | slide   | position (m)             |
|     1 | vertical angle of the pole on the cart        |  -inf |   inf | hinge                              | hinge   | angle (rad)              |
|     2 | linear velocity of the cart                   |  -inf |   inf | slider                             | slide   | velocity (m/s)           |
|     3 | angular velocity of the pole on the cart      |  -inf |   inf | hinge                              | hinge   | angular velocity (rad/s) |

## Rewards

The goal is to keep the inverted pendulum stand upright (within a certain angle limit) for as long as possible - as such, a reward of +1 is given for each timestep that the pole is upright.

The pole is considered upright if:
$|angle| \\leq 0.2$.

and `info` also contains the reward.

## Starting State

The initial position state is $\\mathcal{U} *{[-reset_noise_scale \\times I*{2}, reset_noise_scale \\times I_{2}]}$.
The initial velocity state is $\\mathcal{U} *{[-reset_noise_scale \\times I*{2}, reset_noise_scale \\times I_{2}]}$.

where $\\mathcal{U}$ is the multivariate uniform continuous distribution.

## Episode End

### Termination

The environment terminates when the Inverted Pendulum is unhealthy.
The Inverted Pendulum is unhealthy if any of the following happens:

1. Any of the state space values is no longer finite.
2. The absolute value of the vertical angle between the pole and the cart is greater than 0.2 radians.

### Truncation

The default duration of an episode is 1000 timesteps.

## Arguments

InvertedPendulum provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
These parameters can be applied during `gymnasium.make` in the following way:

```python
import gymnasium as gym
env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
```

| Parameter               | Type      | Default                                | Description                                                                                       |
|-------------------------|-----------|----------------------------------------|---------------------------------------------------------------------------------------------------|
| `xml_file`              | **str**   | `"inverted_pendulum.xml"`              | Path to a MuJoCo model                                                                            |
| `frame_skip`            | **int**   | `2`                                    | Number of MuJoCo simulation steps per `step()`, so $dt = frame_skip \\times$ `model.opt.timestep` |
| `default_camera_config` | **dict**  | `{"trackbodyid": 0, "distance": 2.04}` | Properties of the `mj_camera` used for rendering                                                  |
| `reset_noise_scale`     | **float** | `0.01`                                 | Scale of random perturbations of initial position and velocity (see `Starting State` section)     |

## Version History

* v5:
  - Minimum `mujoco` version is now 2.3.3.
  - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
  - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
  - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
  - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
  - Fixed bug: `healthy_reward` was given on every step (even if the Pendulum is unhealthy), now it is only given if the Pendulum is healthy (not terminated) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/500)).
  - Added `xml_file` argument.
  - Added `reset_noise_scale` argument to set the range of initial states.
  - Added `info["reward_survive"]` which contains the reward.
* v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
* v3: This environment does not have a v3 release. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v2: All continuous control environments now use mujoco-py >= 1.5. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
* v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum).
* v0: Initial versions release.
