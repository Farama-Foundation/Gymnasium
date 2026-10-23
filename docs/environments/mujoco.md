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
MuJoCo was introduced by Todorov, Erez and Tassa in ["MuJoCo: A physics engine for model-based control"](https://doi.org/10.1109/IROS.2012.6386109) (IROS 2012).

The unique dependencies including the MuJoCo simulator for this set of environments can be installed via:

````bash
pip install gymnasium[mujoco]
````

DeepMind acquired MuJoCo in October 2021 and open-sourced it in 2022, making it free for everyone.
Using MuJoCo with Gymnasium requires the framework `mujoco` be installed (this dependency is installed with the above command).
Instructions for installing the MuJoCo engine can be found on their [website](https://mujoco.org) and [GitHub repository](https://github.com/google-deepmind/mujoco).

MuJoCo `v3` environments and older, which relied on the `mujoco-py` framework, were migrated to the `gymnasium-robotics` package starting with `gymnasium` v1.2. For information on using these older versions, please refer to the `gymnasium-robotics` documentation. The original `mujoco-py` framework can be found in its [GitHub repository](https://github.com/openai/mujoco-py/tree/master/mujoco_py).

The dependencies for the old environment versions can be installed via:
````bash
pip install gymnasium_robotics[mujoco-py]
````

There are eleven MuJoCo environments (in roughly increasing complexity):

| Robot                  | Short Description                                                    |
|------------------------|----------------------------------------------------------------------|
| **CartPoles**          |                                                                      |
| InvertedPendulum       | MuJoCo version of the CartPole Environment (with Continuous actions) |
| InvertedDoublePendulum | 2 Pole variation of the CartPole Environment                         |
| **Arms**               |                                                                      |
| Reacher                | 2d arm with the goal of reaching an object                           |
| Pusher                 | 3d arm with the goal of pushing an object to a target location       |
| **2D Runners**         |                                                                      |
| HalfCheetah            | 2d quadruped with the goal of running                                |
| Hopper                 | 2d monoped with the goal of hopping                                  |
| Walker2d               | 2d biped with the goal of walking                                    |
| **Swimmers**           |                                                                      |
| Swimmer                | 2d robot with the goal of swimming                                   |
| **Quadruped**          |                                                                      |
| Ant                    | 3d quadruped with the goal of running                                |
| **Humanoid Bipeds**    |                                                                      |
| Humanoid               | 3d humanoid with the goal of running                                 |
| HumanoidStandup        | 3d humanoid with the goal of standing up                             |

All of these environments are stochastic in terms of their initial state, with uniform or Gaussian noise (depending on the environment) added to a fixed initial state in order to add stochasticity.
The observation spaces of most MuJoCo environments in Gymnasium are built from the positions of the body parts and joints (`mujoco.MjData.qpos`) and their corresponding velocities (`mujoco.MjData.qvel`), flattened and concatenated together (more information in the [MuJoCo Physics State Documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#the-state)).
Some positions (such as the x-coordinate of the torso) are excluded by default, and several environments transform these values (e.g. Reacher and InvertedDoublePendulum use the sine and cosine of joint angles) or add other quantities such as body positions, inertias and contact forces; see the "Observation Space" section of each environment.
<!--
Often some of the first positional elements are omitted from the state space since the reward is calculated based on their values, leaving it up to the algorithm to infer these hidden values indirectly.
-->

Among the Gymnasium environments, this set of environments can be considered as more difficult to solve by policy.

Environments can be configured by changing the `xml_file` argument and/or by tweaking the parameters of their classes.


## Versions
Gymnasium includes the following versions of the environments:

| Version | Simulator       | Notes                                                                                           |
|---------|-----------------|-------------------------------------------------------------------------------------------------|
| `v5`    | `mujoco>=2.3.3` | Recommended (most features, the least bugs)                                                     |
| `v4`    | `mujoco>=2.1.3` | Maintained for reproducibility                                                                  |
| `v3`    | `mujoco-py`     | Migrated to `gymnasium-robotics` (from `gymnasium` v1.2). Deprecated, Kept for reproducibility. |
| `v2`    | `mujoco-py`     | Migrated to `gymnasium-robotics` (from `gymnasium` v1.2). Deprecated, Kept for reproducibility. |

For more information, see the section "Version History" for each environment.

`v1` and older are no longer included in Gymnasium.

### Comparing training performance across versions
The training performance of `v2` and `v3` is identical assuming the same/default arguments were used.

The training performance of `v2`/`v3` and `v4` are not directly comparable because of the change to the newer simulator, but the results for not Ant and not Humanoids are comparable (for more information see [GitHub Comment #1](https://github.com/openai/gym/pull/2595#issuecomment-1099152505) and [GitHub Comment #2](https://github.com/openai/gym/pull/2762#issuecomment-1135362092)).

The Training performance of `v4` and `v5` is different because of the many changes in the environments, but the Half Cheetah and Swimmer exhibit identical behaviour, Pusher and Reacher are close (for more information see [GitHub Issue](https://github.com/Farama-Foundation/Gymnasium/issues/821)).

### Exact reproducibility
Note: The exact behavior of the MuJoCo simulator changes slightly between `mujoco` versions due to floating point operation ordering (more information of their [Documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#reproducibility)), if exact reproducibility is needed besides using the `seed` for experiments the same simulator version should be used.

## Rendering Arguments
All of the MuJoCo Environments besides the general Gymnasium arguments, and environment specific arguments they also take the following arguments for configuring the renderer:

```python
import gymnasium

env = gymnasium.make("Ant-v5", render_mode="rgb_array", width=1280, height=720)
```

| Parameter               | Type                                | Default | Description                                                                                                                                                                                                                                              |
|-------------------------|-------------------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `render_mode`           | **str**                             | `None`  | The modality of the render result. Must be one of `human`, `rgb_array`, `depth_array`, or `rgbd_tuple`. Note that `human` does not return a rendered image, but renders directly to the window                                                           |
| `width`                 | **int**                             | `480`   | The width of the render window                                                                                                                                                                                                                           |
| `height`                | **int**                             | `480`   | The height of the render window                                                                                                                                                                                                                          |
| `camera_id`             | **int \| None**                     | `None`  | The camera ID used for the render window                                                                                                                                                                                                                 |
| `camera_name`           | **str \| None**                     | `None`  | The name of the camera used for the render window (mutually exclusive option with `camera_id`)                                                                                                                                                           |
| `default_camera_config` | **dict[str, float \| int] \| None** | `None`  | The [mjvCamera](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjvcamera) properties                                                                                                                                                 |
| `max_geom`              | **int**                             | `1000`  | Max number of geometrical objects to render (useful for 3rd-party environments)                                                                                                                                                                          |
| `visual_options`        | **dict[int, bool] \| None**         | `None`  | A dictionary with [mjVisual](https://mujoco.readthedocs.io/en/stable/overview.html#mjvisual) flags and value pairs, example `{mujoco.mjtVisFlag.mjVIS_CONTACTPOINT: True, mujoco.mjtVisFlag.mjVIS_CONTACTFORCE: True}` (show contact points and forces). |

### Rendering Backend
The MuJoCo simulator renders images with OpenGL and can use 3 different back ends "glfw" (default), "egl", "osmesa", which can be selected by setting an [environment variable](https://en.wikipedia.org/wiki/Environment_variable).

| Backend  | Environment Variable       | Description                       |
|----------|----------------------------|-----------------------------------|
| `GLFW`   | `MUJOCO_GL=glfw` (default) | Renders with Window System on GPU |
| `EGL`    | `MUJOCO_GL=egl`            | Renders headless on GPU           |
| `OSMESA` | `MUJOCO_GL=osmesa`         | Renders headless on CPU           |

More information of the [MuJoCo/OpenGL documentation](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl).
<!--
## Custom Models
For more complex locomotion robot environments you can use third party models with the environments.
-->
