---
firstpage:
lastpage:
---

# MuJoCo

```{toctree}
:hidden:

mujoco/ant
mujoco/half_cheetah
mujoco/hopper
mujoco/humanoid
mujoco/humanoid_standup
mujoco/inverted_double_pendulum
mujoco/inverted_pendulum
mujoco/pusher
mujoco/reacher
mujoco/swimmer
mujoco/walker2d
```

```{raw} html
   :file: mujoco/list.html
```

MuJoCo stands for Multi-Joint dynamics with Contact. It is a physics engine for facilitating research and development in robotics, biomechanics, graphics and animation, and other areas where fast and accurate simulation is needed.
There is physical contact between the robots and their environment - and MuJoCo attempts at getting realistic physics simulations for the possible physical contact dynamics by aiming for physical accuracy and computational efficiency.

The unique dependencies including the MuJoCo simlator for this set of environments can be installed via:

````bash
pip install gymnasium[mujoco]
````

As of October 2021, DeepMind has acquired MuJoCo and has open-sourced it in 2022, making it free for everyone.
Using MuJoCo with Gymnasium requires the framework `mujoco` be installed (this dependency is installed with the above command).
Instructions for installing the MuJoCo engine can be found on their [website](https://mujoco.org) and [GitHub repository](https://github.com/deepmind/mujoco).

For MuJoCo `v3` environments and older the `mujoco-py` framework is required (`pip install gymnasium[mujoco-py]`) which can be found in the [GitHub repository](https://github.com/openai/mujoco-py/tree/master/mujoco_py).

There are eleven MuJoCo environments (in roughly increasing complexity):
| Robot                  | Short Description                                                    |
| ---------------------- | -------------------------------------------------------------------- |
| **CartPoles**          |                                                                      |
| InvertedPendulum       | MuJoCo version of the CartPole Environment (with Continuous actions) |
| InvertedDoublePendulum | 2 Pole variation of the CartPole Environment                         |
| **Arms**               |                                                                      |
| Reacher                | 2d arm with the goal of reaching an object                           |
| Pusher                 | 3d arm with the goal of pushing an object to a target location       |
| **2D Runners**         |                                                                      |
| HalfCheetah            | 2d quadruped with the goal of running                                |
| Hopper                 | 2d monoped with the goal of hopping                          |
| Walker2d               | 2d biped with the goal of walking                                   |
| **Swimmers**           |                                                                      |
| Swimmer                | 3d robot with the goal of swimming                                   |
| **Quarduped**          |                                                                      |
| Ant                    | 3d quadruped with the goal of running                                |
| **Humanoid Bipeds**    |                                                                      |
| Humanoid               | 3d humanoid with the goal of running                                 |
| HumanoidStandup        | 3d humanoid with the goal of standing up                             |

All of these environments are stochastic in terms of their initial state, with a Gaussian noise added to a fixed initial state in order to add stochasticity.
The state spaces for MuJoCo environments in Gymnasium consist of two parts that are flattened and concatenated together: the position of the body part and joints (`mujoco.MjData.qpos`) and their corresponding velocity (`mujoco.MjData.qvel`) (more information in the [MuJoCo Physics State Documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#physics-state)).
<!--
Often some of the first positional elements are omitted from the state space since the reward is calculated based on their values, leaving it up to the algorithm to infer these hidden values indirectly.
-->

Among the Gymnasium environments, this set of environments can be considered as more difficult to solve by policy.

Environments can be configured by changing the `xml_file` argument and/or by tweaking the parameters of their classes.


## Versions
Gymnasium includes the following versions of the environments:

| Version | Simulator       | Notes                                            |
| ------- | --------------- | ------------------------------------------------ |
| `v5`    | `mujoco=>2.3.3` | Recommended (most features, the least bugs)      |
| `v4`    | `mujoco=>2.1.3` | Maintained for reproducibility                   |
| `v3`    | `mujoco-py`     | Deprecated, Kept for reproducibility (limited support) |
| `v2`    | `mujoco-py`     | Deprecated, Kept for reproducibility (limited support) |

For more information, see the section "Version History" for each environment.

`v1` and older are no longer included in Gymnasium.

### Comparing training performance across versions
The training performance of `v2` and `v3` is identical assuming the same/default arguments were used.

The training performance of `v2`/`v3` and `v4` are not directly comparable because of the change to the newer simulator, but the results for not Ant and not Humanoids are comparable (for more information see [GitHub Comment #1](https://github.com/openai/gym/pull/2595#issuecomment-1099152505) and [GitHub Comment #2](https://github.com/openai/gym/pull/2762#issuecomment-1135362092)).

The Training performance of `v4` and `v5` is different because of the many changes in the environments, but the Half Cheetah and Swimmer exhibits identical behaviour, Pusher and Swimmer are close (for more information see [GitHub Issue](https://github.com/Farama-Foundation/Gymnasium/issues/821)).

### Exact reproducibility
Note: The exact behavior of the MuJoCo simulator changes slightly between `mujoco` versions due to floating point operation ordering (more information of their [Documentation]( https://mujoco.readthedocs.io/en/stable/computation/index.html#reproducibility)), if exact reproducibility is need besides using the `seed` for experiments the same simulator version should be used.

## Rendering Arguments
All of the MuJoCo Environments besides the general Gymnasium arguments, and environment specific arguments they also take the following arguments for configuring the renderer:

```python
env = gymnasium.make("Ant-v5", render_mode="rgb_array", width=1280, height=720)
```

| Parameter                   | Type                              | Default      | Description                               |
| --------------------------- | -------------                     | ------------ | ----------------------------------------- |
| `width`                     | **int**                           | `480`        | The width of the render window            |
| `height`                    | **int**                           | `480`        | The height of the render window           |
| `camera_id`                 |**int \| None**                    | `None`       | The camera ID used for the render window  |
| `camera_name`               |**str \| None**                    | `None`       | The name of the camera used for the render window (mutally exclusive option with `camera_id`) |
| `default_camera_config`     |**dict[str, float \| int] \| None**| `None`       |  The [mjvCamera](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjvcamera) properties |
| `max_geom`                  | **int**                           | `1000`       | Max number of geometrical objects to render (useful for 3rd-party environments) |
| `visual_options`            | **Dict[int, bool]**               | `{}`         | A dictionary with [mjVisual](https://mujoco.readthedocs.io/en/stable/overview.html#mjvisual) flags and value pairs, example `{mujoco.mjtVisFlag.mjVIS_CONTACTPOINT: True, mujoco.mjtVisFlag.mjVIS_CONTACTFORCE: True}` (show contact points and forces). |



<!--
## Custom Models
For more complex locomotion robot environments you can use third party models with the environments.
-->
