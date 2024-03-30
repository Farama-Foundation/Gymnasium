---
layout: blog
short_title: "Gymnasium/MuJoCo-v5 Environments"
subtitle: "New Features of Gymnasium/MuJoCo-v5 Environments, and how to load third-party models"
title: "Release Gymnasium/MuJoCo-v5 Environments"
date: "2024-0?-??"
excerpt: ""
Author: Kallinteris Andreas
thumbnail: 
image: 
read_time: ???
---


# Introduction
`Gymnasium/MuJoCo` is a set of reinforcement learning environments where the goal is to make a robot locomote as fast as possible.

The version 0 of `Gymnasium/MuJoCo` was introduced in `gym==0.0.1` 2016,
followed shortly after by version 1 (`gym==0.1.0` in 2016) which addressed several configuration errors.
Version 2 (`Gym==0.9.5` in 2018), which brought major backend improvements using the `mujoco-py=>1.5` simulator. 
Version 3 (`gym==0.12.0` in 2019) offers increased customization options, enabling the user to modify parts of the environment, such as the reward function, and make slight adjustments to the robot model.
Version 4 (`gym==0.24.0` in 2020)  ported the environments to the new `mujoco>=2.2.0` simulator. However it removed the capability of slightly modifying the environment.

`Gymnasium/MuJoCo` environments such as Hopper, Ant, and Humanoid have been utilized in thousands of RL research papers. They have been a cornerstone in the development and evaluation RL methods, as they have been a standard set of "hard to solve" robotic environments.

However, as RL methods continue the need for more complex robotic environments to evaluate them becomes apparent, for example state-of-the-art training algorithms like `TD3` and `SAC` can "solve" Ant and Humanoid.  

With the release of `gymnasium==1.0.0`, we introduce the new v5 version of the Gymnasium/MuJoCo environments. This version is highly customizable, has fewer bugs and is faster.
`Gymnasium/MuJoCo-v5` environments/framework is available with `gymnasium==1.0.0`
```sh
pip install gymnasium>=1.0.0
```

# Key features:
- Add Support for loading third-party MuJoCo models, such as realistic robot models.
-- e.g. [MuJoCo Menagerie](https://github.com/deepmind/mujoco_menagerie).
<!--
[MyoSim](https://github.com/facebookresearch/myosuite)
-->

- 80 total enhancements, notably:
-- Performance: Improves training performance by removing a considerable amount of constant 0 observations from the observation spaces of the `Ant`, `Humanoid`, `HumanoidStandup` (5-7% faster training with `SB3/PPO`-`pytorch`) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/204)).
-- Customizability: Added new arguments for all the environments & restored removed arguments in the `v3 → v4` transition.
-- QOL: Added new fields in `info` inlcuding all reward components and non-observable state and `reset()` now returns `info` which includes non observable state elements.

- 23 bugs fixed, notably:
-- In Ant, Hopper, Humanoid, InvertedDoublePendulum, InvertedPendulum, Walker2d: `healthy_reward`, was being given on every step (even if the robot is unhealthy), now it is only given when the robot is healthy,  resulting in faster learning transience (for more information on the performance impact of this change, check the related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/526)).
-- In Ant & Humanoid: The `contact_cost` being constantly 0 (for more information on the performance impact of this change, check the related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/504)).
-- In Reacher & Pusher the reward function was calculated baed the previous state not the current state (for more information on the performance impact of this change, check the related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/821)).
-- Fixed several `info` fields.
-- Walker2D: Both feet now have `friction==1.9`, previously the right foot had `friction==0.9` and the left foot had `friction==1.9` (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/477)).

- Generally improved documentation.



# Using third-party MuJoCo robot models, examples:
For those looking for more computationally complex real-world robot MuJoCo models, `v5` now supports the use of these models with little effort.
Here in this blog, we will use robot MuJoCo models from the [MuJoCo Menagerie](https://github.com/deepmind/mujoco_menagerie) project.

For quadruped robots, we recommend using the `Ant-v5` framework:
```py
env = gymnasium.make('Ant-v5', xml_file="MY_MUJOCO_MODEL.xml", ...)
```
For bipedal robots, we recommend using the `Humanoid-v5` framework:
```py
env = gymnasium.make('Humanoid-v5', xml_file="MY_MUJOCO_MODEL.xml", ...)
```
For swimmer and crawler robots, we recommend using the `Swimmer-v5` framework:
```py
env = gymnasium.make('Swimmer-v5', xml_file="MY_MUJOCO_MODEL.xml", ...)
```

But we will need to change some arguments to specify the behavior we want, the most commonly changed arguments are:
- `xml_file`: Path to the MuJoCo robot.
- `frame_skip`: to set the duration of a time step (`dt`) (recommended range of `dt` is $\[0.01, 0.1\]$). 
- `ctrl_cost_weight`: set it according to the needs of the robot, we can set it to `0` at first for prototyping and increase it as needed.
- `healthy_z_range`: set it according to the height of the robot.
For more information on all the arguments, see the documentation page of `Humanoid`, `Ant` & respectively.

## Using `Ant` framework to create an RL environment for a Quadruped robot

### Example [anybotics_anymal_b](https://github.com/deepmind/mujoco_menagerie/blob/main/anybotics_anymal_b/README.md)
```py
env = gymnasium.make('Ant-v5', xml_file='./mujoco_menagerie/anybotics_anymal_b/scene.xml', ctrl_cost_weight=0.001, healthy_z_range=(0.48, 0.68), render_mode='human')
```
Here all we have to do is to change the `xml_file` argument, and set the `healthy_z_range`, because the robot has a different height than the default `Ant` robot, in general we will have to change the `healthy_z_range` to fit the robot.

<iframe id="odysee-iframe" width="560" height="315" src="https://odysee.com/$/embed/@Kallinteris-Andreas:7/ANYmal_B_trained_using_SAC_on_gymnasium_mujoco-v5_framework:1?r=6fn5jA9uZQUZXGKVpwtqjz1eyJcS3hj3" allowfullscreen></iframe>
 
 ### Example [Unitree Go1](https://github.com/deepmind/mujoco_menagerie/blob/main/unitree_go1/README.md)
```py
env = gym.make('Ant-v5', xml_file='./mujoco_menagerie/unitree_go1/scene.xml', healthy_z_range=(0.195, 0.75), ctrl_cost_weight=0.05)
```
<iframe id="odysee-iframe" width="560" height="315" src="https://odysee.com/$/embed/@Kallinteris-Andreas:7/Unitree_Go1_trained_using_SAC_on_gymnasium_mujoco-v5_framework:5?r=6fn5jA9uZQUZXGKVpwtqjz1eyJcS3hj3" allowfullscreen></iframe>

## Using `Humanoid` framework to create an RL environment for a Bipeds robot


### Example [Robotis OP3](https://github.com/deepmind/mujoco_menagerie/blob/main/robotis_op3/README.md)
```py
env = gym.make('Humanoid-v5', xml_file='~/mujoco_menagerie/robotis_op3/scene.xml', healthy_z_range=(0.275, 0.5), include_cinert_in_observation=False, include_cvel_in_observation=False, include_qfrc_actuator_in_observation=False, include_cfrc_ext_in_observation=False, ctrl_cost_weight=0, contact_cost_weight=0)
```

<iframe id="odysee-iframe" width="560" height="315" src="https://odysee.com/$/embed/@Kallinteris-Andreas:7/Robotis_OP3_trained_using_PPO_on_gymnasium_mujoco-v5_framework:d?r=6fn5jA9uZQUZXGKVpwtqjz1eyJcS3hj3" allowfullscreen></iframe>

For more detailed tutorial check the tutorial for [loading quadruped models](https://gymnasium.farama.org/main/tutorials/gymnasium_basics/load_quadruped_model/).



# Full Changelog
For more information about the development of the `Gymnasium/MuJoCo-v5` environments and a complete changelog, check the [GitHub PR](https://github.com/Farama-Foundation/Gymnasium/pull/572).
