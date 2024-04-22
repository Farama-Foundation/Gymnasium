---
layout: blog
short_title: "Gymnasium/MuJoCo-v5 Environments"
subtitle: "New Features of Gymnasium/MuJoCo-v5 Environments and How to Load Third-Party Models"
title: "Release Gymnasium/MuJoCo-v5 Environments"
date: "2024-0?-??"
excerpt: ""
Author: Kallinteris Andreas
thumbnail: 
image: 
read_time: 4 minutes
---

# Introduction
`Gymnasium/MuJoCo` is a set of robotics based reinforcement learning environments using the [mujoco](https://mujoco.org/) physics engine with various different goals for the robot to learn: standup, run quickly, move an arm to a point.

## A quick history of MuJoCo environments 
Originally introduced as version 0 in `gym==0.0.1` way back in 2016, this was shortly followed by version 1 (`gym=0.1.0`) to address several configuration errors. 
In 2018, version 2 in `gym=0.9.5` was released, which brought major backend improvements using `mujoco-py=>1.5` simulator. 
Version 3 (`gym=0.12.0` in 2018) offered increased customization options, enabling users to modify parts of the environment such as the reward function and slight adjustments to the robot model. 
With Google-DeepMind buying MuJoCo, open sourcing the code and releasing a dedicated python module (`mujoco`), version 4 ports the environments to the new `mujoco>=2.2.0` simulator however removed the capability to slightly modify the environment. 

## MuJoCo v5
Over time, the MuJoCo environments have become standard testing environments in RL, used in hundreds if not thousands of academic papers at this point. They have provided a standard set of difficult-to-solve robotic environments, and a cornerstone in the development and evaluation of RL methods.

However, as RL methods continue to improve, the necessity for more complex robotic environments to evaluate them has become evident with state-of-the-art training algorithms, such as [TD3](https://arxiv.org/pdf/1802.09477.pdf) and [SAC](https://arxiv.org/pdf/1801.01290.pdf), being able to solve even the more complex of the MuJoCo problems.  

We are pleased to announce that with`gymnasium==1.0.0`, we introduce the new version 5 version of the Gymnasium/MuJoCo environments with significantly increased customizability, bugs fixes and is faster.
```sh
pip install "gymnasium[mujoco]>=1.0.0"
```

```python
import gymnasium as gym

env = gym.make("Humanoid-v5")
```

## Key features:
- Add support for loading third-party MuJoCo models, including realistic robot models including [MuJoCo Menagerie](https://github.com/deepmind/mujoco_menagerie).

- 80 enhancements, notably:
  - Performance: Improves training performance by removing a considerable amount of constant 0 observations from the observation spaces of the `Ant`, `Humanoid` and `HumanoidStandup`. This results in 5-7% faster training with `SB3/PPO` using `pytorch` (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/204)).
  - Customizability: Added new arguments for all the environments and restored the removed arguments from the `v3 → v4` transition.
  - Quality of life: Added new fields to `info` that include all reward components and non-observable state, and `reset()` now returns `info` which includes non-observable state elements.

- 24 bugs fixes, notably:
  - Walker2D: Both feet now have `friction==1.9`, previously the right foot had `friction==0.9` and the left foot had `friction==1.9` (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/477)).
  - Pusher: Fixed the issue of the object being lighter than air cause the cylinder physics unexpected behaviour. We have increased the weight to a more realistic value (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/950)).
  - In Ant, Hopper, Humanoid, InvertedDoublePendulum, InvertedPendulum, Walker2d: `healthy_reward`, was previously given on every step (even if the robot was unhealthy), now it is only given when the robot is healthy, resulting in faster learning. For further details about the performance impact of this change, see the related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/526)
  - In Ant and Humanoid, the `contact_cost` reward component was constantly 0. For more information on the performance impact of this change, please check the related [GitHub issue #1](https://github.com/Farama-Foundation/Gymnasium/issues/504), [GitHub issue #2](https://github.com/Farama-Foundation/Gymnasium/issues/214).
  - In Reacher and  Pusher, the reward function was calculated based on the previous state not the current state. For further information on the performance impact of this change, see the related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/821).
  - Fixed several `info` fields.

- Generally improved documentation to explain the observation, action and reward functions in more detail. 

## Example using a third-party MuJoCo robot models:
For those looking for more complex real-world robot MuJoCo models, `v5` now supports the use of these models. Below, we show how this can be achieved using models from the [MuJoCo Menagerie](https://github.com/deepmind/mujoco_menagerie) project.

Depending on the robot type, we recommend using different environment models: for quadruped → `Ant-v5`, bipedal → `Humanoid-v5` and swimmer / crawler robots → `Swimmer-v5`. 

However, it will be necessary to modify certain arguments in order to specify the desired behavior. The most commonly changed arguments are:
- `xml_file`: Path to the MuJoCo robot.
- `frame_skip`: to set the duration of a time step (`dt`) (recommended range of `dt` is $\[0.01, 0.1\]$). 
- `ctrl_cost_weight`: set it according to the needs of the robot, we can set it to `0` at first for prototyping and increase it as needed.
- `healthy_z_range`: set it according to the height of the robot.
For more information on all the arguments, see the documentation pages of `Humanoid`, `Ant` and respectively.

### Example [anybotics_anymal_b](https://github.com/deepmind/mujoco_menagerie/blob/main/anybotics_anymal_b/README.md)
```py
env = gymnasium.make('Ant-v5', xml_file='./mujoco_menagerie/anybotics_anymal_b/scene.xml', ctrl_cost_weight=0.001, healthy_z_range=(0.48, 0.68), render_mode='human')
```

Here all we have to do is change the `xml_file` argument, and set the `healthy_z_range`, because the robot has a different height than the default `Ant` robot.  In general, we will have to change the `healthy_z_range` to fit the robot.

<iframe id="odysee-iframe" width="560" height="315" src="https://odysee.com/$/embed/@Kallinteris-Andreas:7/ANYmal_B_trained_using_SAC_on_gymnasium_mujoco-v5_framework:1?r=6fn5jA9uZQUZXGKVpwtqjz1eyJcS3hj3" allowfullscreen></iframe>
 
 ### Example [Unitree Go1](https://github.com/deepmind/mujoco_menagerie/blob/main/unitree_go1/README.md)
```py
env = gymnasium.make('Ant-v5', xml_file='./mujoco_menagerie/unitree_go1/scene.xml', healthy_z_range=(0.195, 0.75), ctrl_cost_weight=0.05)
```

<iframe id="odysee-iframe" width="560" height="315" src="https://odysee.com/$/embed/@Kallinteris-Andreas:7/Unitree_Go1_trained_using_SAC_on_gymnasium_mujoco-v5_framework:5?r=6fn5jA9uZQUZXGKVpwtqjz1eyJcS3hj3" allowfullscreen></iframe>

### Example [Robotis OP3](https://github.com/deepmind/mujoco_menagerie/blob/main/robotis_op3/README.md)
```py
env = gymnasium.make('Humanoid-v5', xml_file='~/mujoco_menagerie/robotis_op3/scene.xml', healthy_z_range=(0.275, 0.5), include_cinert_in_observation=False, include_cvel_in_observation=False, include_qfrc_actuator_in_observation=False, include_cfrc_ext_in_observation=False, ctrl_cost_weight=0, contact_cost_weight=0)
```
<iframe id="odysee-iframe" width="560" height="315" src="https://odysee.com/$/embed/@Kallinteris-Andreas:7/Robotis_OP3_trained_using_PPO_on_gymnasium_mujoco-v5_framework:d?r=6fn5jA9uZQUZXGKVpwtqjz1eyJcS3hj3" allowfullscreen></iframe>

For a more detailed tutorial, see [loading quadruped models](https://gymnasium.farama.org/main/tutorials/gymnasium_basics/load_quadruped_model/).

# Full Changelog
For more information about the development of the `Gymnasium/MuJoCo-v5` environments and a complete changelog, check the [GitHub PR](https://github.com/Farama-Foundation/Gymnasium/pull/572).