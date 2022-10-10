---
firstpage:
lastpage:
---

## MuJoCo

```{toctree}
:hidden:

ant
half_cheetah
hopper
humanoid_standup
humanoid
inverted_double_pendulum
inverted_pendulum
reacher
swimmer
walker2d
```

```{raw} html
   :file: mujoco/list.html
```

MuJoCo stands for Multi-Joint dynamics with Contact. It is a physics engine for facilitating research and development in robotics, biomechanics, graphics and animation, and other areas where fast and accurate simulation is needed.

The unique dependencies for this set of environments can be installed via:

````bash
pip install gymnasium[mujoco]
````

These environments also require that the MuJoCo engine be installed. As of October 2021 DeepMind has acquired MuJoCo and is open-sourcing it in 2022, making it free for everyone. Instructions on installing the MuJoCo engine can be found on their [website](https://mujoco.org)](https://mujoco.org) and [GitHub repository](https://github.com/deepmind/mujoco). Using MuJoCo with Gymnasium also requires that the framework `mujoco-py` be installed, which can be found in the [GitHub](https://github.com/openai/mujoco-py/tree/master/mujoco_py) repository](https://github.com/openai/mujoco-py/tree/master/mujoco_py) (this dependency is installed with the above command).

There are ten Mujoco environments: Ant, HalfCheetah, Hopper, Humanoid, HumanoidStandup, IvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, and Walker. All of these environments are stochastic in terms of their initial state, with a Gaussian noise added to a fixed initial state in order to add stochasticity. The state spaces for MuJoCo environments in Gymnasium consist of two parts that are flattened and concatenated together: a position of a body part ('*mujoco-py.mjsim.qpos*') or joint and its corresponding velocity ('*mujoco-py.mjsim.qvel*'). Often, some of the first positional elements are omitted from the state space since the reward is calculated based on their values, leaving it up to the algorithm to infer those hidden values indirectly.

Among Gymnasium environments, this set of environments can be considered as more difficult ones to solve by a policy.

Environments can be configured by changing the XML files or by tweaking the parameters of their classes.