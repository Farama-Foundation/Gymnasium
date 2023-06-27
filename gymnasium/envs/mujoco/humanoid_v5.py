__credits__ = ["Kallinteris-Andreas"]

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    This environment is based on the environment introduced by Tassa, Erez and Todorov
    in ["Synthesis and stabilization of complex behaviors through online trajectory optimization"](https://ieeexplore.ieee.org/document/6386025).
    The 3D bipedal robot is designed to simulate a human. It has a torso (abdomen) with a pair of
    legs and arms. The legs each consist of three body parts, and the arms 2 body parts (representing the knees and
    elbows respectively). The goal of the environment is to walk forward as fast as possible without falling over.


    ## Action Space
    The action space is a `Box(-1, 1, (17,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
    |-----|----------------------|---------------|----------------|---------------------------------------|-------|------|
    | 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4 | 0.4 | abdomen_y                   | hinge | torque (N m) |
    | 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4 | 0.4 | abdomen_z                   | hinge | torque (N m) |
    | 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4 | 0.4 | abdomen_x                   | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4 | 0.4 | right_hip_x (right_thigh)   | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4 | 0.4 | right_hip_z (right_thigh)   | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4 | 0.4 | right_hip_y (right_thigh)   | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4 | 0.4 | right_knee                  | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4 | 0.4 | left_hip_x (left_thigh)     | hinge | torque (N m) |
    | 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4 | 0.4 | left_hip_z (left_thigh)     | hinge | torque (N m) |
    | 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4 | 0.4 | left_hip_y (left_thigh)     | hinge | torque (N m) |
    | 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -0.4 | 0.4 | left_knee                   | hinge | torque (N m) |
    | 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -0.4 | 0.4 | right_shoulder1             | hinge | torque (N m) |
    | 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -0.4 | 0.4 | right_shoulder2             | hinge | torque (N m) |
    | 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -0.4 | 0.4 | right_elbow                 | hinge | torque (N m) |
    | 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -0.4 | 0.4 | left_shoulder1              | hinge | torque (N m) |
    | 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -0.4 | 0.4 | left_shoulder2              | hinge | torque (N m) |
    | 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -0.4 | 0.4 | left_elbow                  | hinge | torque (N m) |


    ## Observation Space
    Observations consist of positional values of different body parts of the Humanoid,
    followed by the velocities of those individual parts (their derivatives) with all the
    positions ordered before all the velocities.

    By default, observations do not include the x- and y-coordinates of the torso. These may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will be a `Box(-Inf, Inf, (350,), float64)` where the first two observations
    represent the x- and y-coordinates of the torso.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x- and y-coordinates
    will be returned in `info` with keys `"x_position"` and `"y_position"`, respectively.

    However, by default, the observation is a `Box(-Inf, Inf, (348,), float64)`. The elements correspond to the following:

    | Num | Observation                                                                                                     | Min  | Max | Name (in corresponding XML file) | Joint | Unit                       |
    | --- | --------------------------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | -------------------------- |
    | 0   | z-coordinate of the torso (centre)                                                                              | -Inf | Inf | root                             | free  | position (m)               |
    | 1   | x-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
    | 2   | y-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
    | 3   | z-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
    | 4   | w-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
    | 5   | z-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_z                        | hinge | angle (rad)                |
    | 6   | y-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_y                        | hinge | angle (rad)                |
    | 7   | x-angle of the abdomen (in pelvis)                                                                              | -Inf | Inf | abdomen_x                        | hinge | angle (rad)                |
    | 8   | x-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_x                      | hinge | angle (rad)                |
    | 9   | z-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_z                      | hinge | angle (rad)                |
    | 10  | y-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_y                      | hinge | angle (rad)                |
    | 11  | angle between right hip and the right shin (in right_knee)                                                      | -Inf | Inf | right_knee                       | hinge | angle (rad)                |
    | 12  | x-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_x                       | hinge | angle (rad)                |
    | 13  | z-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_z                       | hinge | angle (rad)                |
    | 14  | y-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_y                       | hinge | angle (rad)                |
    | 15  | angle between left hip and the left shin (in left_knee)                                                         | -Inf | Inf | left_knee                        | hinge | angle (rad)                |
    | 16  | coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder1                  | hinge | angle (rad)                |
    | 17  | coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder2                  | hinge | angle (rad)                |
    | 18  | angle between right upper arm and right_lower_arm                                                               | -Inf | Inf | right_elbow                      | hinge | angle (rad)                |
    | 19  | coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder1                   | hinge | angle (rad)                |
    | 20  | coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder2                   | hinge | angle (rad)                |
    | 21  | angle between left upper arm and left_lower_arm                                                                 | -Inf | Inf | left_elbow                       | hinge | angle (rad)                |
    | 22  | x-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
    | 23  | y-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
    | 24  | z-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
    | 25  | x-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | anglular velocity (rad/s)  |
    | 26  | y-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | anglular velocity (rad/s)  |
    | 27  | z-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | anglular velocity (rad/s)  |
    | 28  | z-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_z                        | hinge | anglular velocity (rad/s)  |
    | 29  | y-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_y                        | hinge | anglular velocity (rad/s)  |
    | 30  | x-coordinate of angular velocity of the abdomen (in pelvis)                                                     | -Inf | Inf | abdomen_x                        | hinge | aanglular velocity (rad/s) |
    | 31  | x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_x                      | hinge | anglular velocity (rad/s)  |
    | 32  | z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_z                      | hinge | anglular velocity (rad/s)  |
    | 33  | y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_y                      | hinge | anglular velocity (rad/s)  |
    | 34  | angular velocity of the angle between right hip and the right shin (in right_knee)                              | -Inf | Inf | right_knee                       | hinge | anglular velocity (rad/s)  |
    | 35  | x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_x                       | hinge | anglular velocity (rad/s)  |
    | 36  | z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_z                       | hinge | anglular velocity (rad/s)  |
    | 37  | y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_y                       | hinge | anglular velocity (rad/s)  |
    | 38  | angular velocity of the angle between left hip and the left shin (in left_knee)                                 | -Inf | Inf | left_knee                        | hinge | anglular velocity (rad/s)  |
    | 39  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder1                  | hinge | anglular velocity (rad/s)  |
    | 40  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder2                  | hinge | anglular velocity (rad/s)  |
    | 41  | angular velocity of the angle between right upper arm and right_lower_arm                                       | -Inf | Inf | right_elbow                      | hinge | anglular velocity (rad/s)  |
    | 42  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder1                   | hinge | anglular velocity (rad/s)  |
    | 43  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder2                   | hinge | anglular velocity (rad/s)  |
    | 44  | angular velocity of the angle between left upper arm and left_lower_arm                                         | -Inf | Inf | left_elbow                       | hinge | anglular velocity (rad/s)  |
    | excluded | x-coordinate of the torso (centre)                                                                         | -Inf | Inf | root                             | free  | position (m)               |
    | excluded | y-coordinate of the torso (centre)                                                                         | -Inf | Inf | root                             | free  | position (m)               |

    Additionally, after all the positional and velocity based values in the table,
    the observation contains (in order):
    - *cinert:* Mass and inertia of a single rigid body relative to the center of mass
    (this is an intermediate result of transition). It has shape 13*10 (*nbody * 10*)
    and hence adds to another 130 elements in the state space.
    - *cvel:* Center of mass based velocity. It has shape 13 * 6 (*nbody * 6*) and hence
    adds another 78 elements in the state space
    - *qfrc_actuator:* Constraint force generated as the actuator force. This has shape
    `(17,)`  *(nv * 1)* and hence adds another 17 elements to the state space.
    - *cfrc_ext:* This is the center of mass based external force on the body.  It has shape
    13 * 6 (*nbody * 6*) and hence adds to another 78 elements in the state space.
    where *nbody* stands for the number of bodies in the robot and *nv* stands for the
    number of degrees of freedom (*= dim(qvel)*)

    The body parts are:

    | id (for `v2`, `v3`, `v4)` | id (for `v5`) | body part |
    | ---|  ---   |  ------------  |
    | 0  |excluded| worldbody (note: all values are constant 0) |
    | 1  | 0      | torso |
    | 2  | 1      | lwaist |
    | 3  | 2      | pelvis |
    | 4  | 3      | right_thigh |
    | 5  | 4      | right_sin |
    | 6  | 5      | right_foot |
    | 7  | 6      | left_thigh |
    | 8  | 7      | left_sin |
    | 9  | 8      | left_foot |
    | 10 | 9      | right_upper_arm |
    | 11 | 10     | right_lower_arm |
    | 12 | 11     | left_upper_arm |
    | 13 | 12     | left_lower_arm |

    The joints are:

    | id (for `v2`, `v3`, `v4)` | id (for `v5`) | joint |
    | ---|  ---   |  ------------  |
    | 0  |excluded| root (note: all values are constant 0) |
    | 1  |excluded| root (note: all values are constant 0) |
    | 2  |excluded| root (note: all values are constant 0) |
    | 3  |excluded| root (note: all values are constant 0) |
    | 4  |excluded| root (note: all values are constant 0) |
    | 5  |excluded| root (note: all values are constant 0) |
    | 6  | 0      | abdomen_z |
    | 7  | 1      | abdomen_y |
    | 8  | 2      | abdomen_x |
    | 9  | 3      | right_hip_x |
    | 10 | 4      | right_hip_z |
    | 11 | 5      | right_hip_y |
    | 12 | 6      | right_knee |
    | 13 | 7      | left_hip_x |
    | 14 | 8      | left_hiz_z |
    | 15 | 9      | left_hip_y |
    | 16 | 10     | left_knee |
    | 17 | 11     | right_shoulder1 |
    | 18 | 12     | right_shoulder2 |
    | 19 | 13     | right_elbow|
    | 20 | 14     | left_shoulder1 |
    | 21 | 15     | left_shoulder2 |
    | 22 | 16     | left_elfbow |

    The (x,y,z) coordinates are translational DOFs while the orientations are rotational
    DOFs expressed as quaternions. One can read more about free joints on the
    [Mujoco Documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

    **Note:** Humanoid-v4 environment no longer has the following contact forces issue.
    If using previous Humanoid versions from v4, there have been reported issues that using a Mujoco-Py version > 2.0
    results in the contact forces always being 0. As such we recommend to use a Mujoco-Py
    version < 2.0 when using the Humanoid environment if you would like to report results
    with contact forces (if contact forces are not used in your experiments, you can use
    version > 2.0).


    ## Rewards
    The reward consists of three parts:
    - *healthy_reward*:
    Every timestep that the Humanoid is alive (see section Episode Termination for definition),
    it gets a reward of fixed value `healthy_reward`.
    - *forward_reward*:
    A reward of moving forward,
    this reward would be positive if the Humanoid moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the center of mass ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions which is dependent on the `frame_skip` parameter (default is 5),
    and `frametime` which is 0.001 - making the default $dt = 5 \times 0.003 = 0.015$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1.25$).
    The calculation for the center of mass is defined in the `.py` file for the Humanoid.
    - *ctrl_cost*:
    A negative reward for penalizing the Humanoid if it takes actions that are too large.
    $w_{control} \times \\|action\\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $0.1$).
    If there are *nu* actuators/controls, then the control has shape  `nu x 1`.
    - *contact_cost*:
    A negative reward for penalizing the Humanoid if the external contact forces are too large.
    $w_{contact} \times clamp(contact\\_cost\\_range, \\|F_{contact}\\|_2^2)$, where
    $w_{contact}$ is `contact_cost_weight` (default is $5\times10^{-7}$),
    $F_{contact}$ are the external contact forces (see `cfrc_ext` section on observation).

    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost*
    and `info` will also contain the individual reward terms.

    Note: in `v4` the total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*


    ## Starting State
    All observations start in state
    (0.0, 0.0,  1.4, 1.0, 0.0  ... 0.0) with a uniform noise in the range
    of [-`reset_noise_scale`, `reset_noise_scale`] added to the positional and velocity values (values in the table)
    for stochasticity. Note that the initial z coordinate is intentionally
    selected to be high, thereby indicating a standing up humanoid. The initial
    orientation is designed to make it face forward as well.


    ## Episode End
    #### Termination
    If `terminate_when_unhealthy is True` (which is the default), the environment terminates when the Humanoid is unhealthy.
    The Humanoid is said to be unhealthy if any of the following happens:

    1. The z-position of the torso (the height) is no longer contained in `healthy_z_range`.

    #### Truncation
    The maximum duration of an episode is 1000 timesteps.


    ## Arguments
    `gymnasium.make` takes additional arguments such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.

    ```python
    import gymnasium as gym
    env = gym.make('Humanoid-v5', ctrl_cost_weight=0.1, ....)
    ```

    | Parameter                                    | Type      | Default          | Description                                                                                                                                                               |
    | -------------------------------------------- | --------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"humanoid.xml"` | Path to a MuJoCo model                                                                                                                                                    |
    | `forward_reward_weight`                      | **float** | `1.25`           | Weight for _forward_reward_ term (see section on reward)                                                                                                                  |
    | `ctrl_cost_weight`                           | **float** | `0.1`            | Weight for _ctrl_cost_ term (see section on reward)                                                                                                                       |
    | `contact_cost_weight`                        | **float** | `5e-7`           | Weight for _contact_cost_ term (see section on reward)                                                                                                                    |
    | `contact_cost_range`                         | **float** | `(-np.inf, 10.0) | Clamps the _contact_cost_ term (see section on reward)                                                                                                                    |
    | `healthy_reward`                             | **float** | `5.0`            | Weight for _healthy_reward_ term (see section on reward)                                                                                                                    |
    | `terminate_when_unhealthy`                   | **bool**  | `True`           | If true, issue a done signal if the z-coordinate of the torso is no longer in the `healthy_z_range`                                                                       |
    | `healthy_z_range`                            | **tuple** | `(1.0, 2.0)`     | The humanoid is considered healthy if the z-coordinate of the torso is in this range                                                                                      |
    | `reset_noise_scale`                          | **float** | `1e-2`           | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                            |
    | `exclude_current_positions_from_observation` | **bool**  | `True`           | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |
    | `include_cinert_in_observation`              | **bool**  | `True`           | Whether to include *cinert* elements in the observations.                                                                                                                 |
    | `include_cvel_in_observation`                | **bool**  | `True`           | Whether to include *cvel* elements in the observations.                                                                                                                   |
    | `include_qfrc_actuator_in_observation`       | **bool**  | `True`           | Whether to include *qfrc_actuator* elements in the observations.                                                                                                          |
    | `include_cfrc_ext_in_observation`            | **bool**  | `True`           | Whether to include *cfrc_ext* elements in the observations.                                                                                                               |

    ## Version History
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco-py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file="humanoid.xml",
        frame_skip=5,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        contact_cost_range=(-np.inf, 10.0),
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        include_cinert_in_observation=True,
        include_cvel_in_observation=True,
        include_qfrc_actuator_in_observation=True,
        include_cfrc_ext_in_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            contact_cost_range,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cinert": self.data.cinert[1:].size * include_cinert_in_observation,
            "cvel": self.data.cvel[1:].size * include_cvel_in_observation,
            "qfrc_actuator": (self.data.qvel.size - 6)
            * include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
            "ten_lenght": 0,
            "ten_velocity": 0,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        # TODO remove after validation
        assert terminated == (
            not self.is_healthy if self._terminate_when_unhealthy else False
        )
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:].flat.copy()
        else:
            com_inertia = np.array([])
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:].flat.copy()
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator[6:].flat.copy()
        else:
            actuator_forces = np.array([])
        if self._include_cfrc_ext_in_observation is True:
            external_contact_forces = self.data.cfrc_ext[1:].flat.copy()
        else:
            external_contact_forces = np.array([])

        # TODO remove after validation
        assert (self.data.cinert[0].flat.copy() == 0).all()
        assert (self.data.cvel[0].flat.copy() == 0).all()
        assert (self.data.qfrc_actuator[:6].flat.copy() == 0).all()
        assert (self.data.cfrc_ext[0].flat.copy() == 0).all()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - costs
        terminated = self.terminated
        info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_lenght": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_lenght": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
