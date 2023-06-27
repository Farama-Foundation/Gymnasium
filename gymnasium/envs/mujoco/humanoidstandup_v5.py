__credits__ = ["Kallinteris-Andreas"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8925)),
    "elevation": -20.0,
}


class HumanoidStandupEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    This environment is based on the environment introduced by Tassa, Erez and Todorov
    in ["Synthesis and stabilization of complex behaviors through online trajectory optimization"](https://ieeexplore.ieee.org/document/6386025).
    The 3D bipedal robot is designed to simulate a human. It has a torso (abdomen) with a
    pair of legs and arms. The legs each consist of two links, and so the arms (representing the
    knees and elbows respectively). The environment starts with the humanoid laying on the ground,
    and then the goal of the environment is to make the humanoid standup and then keep it standing
    by applying torques on the various hinges.


    ## Action Space
    The agent take a 17-element vector for actions.

    The action space is a continuous `(action, ...)` all in `[-1, 1]`, where `action`
    represents the numerical torques applied at the hinge joints.

    | Num | Action                                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | ---------------------------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_y                        | hinge | torque (N m) |
    | 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_z                        | hinge | torque (N m) |
    | 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_x                        | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4        | 0.4         | right_hip_x (right_thigh)        | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4        | 0.4         | right_hip_z (right_thigh)        | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4        | 0.4         | right_hip_y (right_thigh)        | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4        | 0.4         | right_knee                       | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4        | 0.4         | left_hip_x (left_thigh)          | hinge | torque (N m) |
    | 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4        | 0.4         | left_hip_z (left_thigh)          | hinge | torque (N m) |
    | 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4        | 0.4         | left_hip_y (left_thigh)          | hinge | torque (N m) |
    | 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -0.4        | 0.4         | left_knee                        | hinge | torque (N m) |
    | 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -0.4        | 0.4         | right_shoulder1                  | hinge | torque (N m) |
    | 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -0.4        | 0.4         | right_shoulder2                  | hinge | torque (N m) |
    | 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -0.4        | 0.4         | right_elbow                      | hinge | torque (N m) |
    | 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -0.4        | 0.4         | left_shoulder1                   | hinge | torque (N m) |
    | 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -0.4        | 0.4         | left_shoulder2                   | hinge | torque (N m) |
    | 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -0.4        | 0.4         | left_elbow                       | hinge | torque (N m) |


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
    - *uph_cost*:
    A reward for moving upward (in an attempt to stand up).
    This is not a relative reward which measures how much upward it has moved from the last timestep,
    but it is an absolute reward which measures how much upward the Humanoid has moved overall.
    It is measured as $weight_{uph} \times (z_{after action} - 0)/dt$,
    where $z_{after action}$ is the z coordinate of the torso after taking an action,
    and *dt* is the time between actions and is dependent on the `frame_skip` parameter
    (default is 5), where the frametime is 0.003 - making the default *dt = 5 * 0.003 = 0.015*.
    and $weight_{uph}$ is `uph_cost_weight`.
    - *quad_ctrl_cost*:
    A negative reward for penalizing the Humanoid if it takes actions that are too large.
    $w_{quad_control} \times \\|action\\|_2^2$,
    where $w_{quad_control}$ is `ctrl_cost_weight` (default is $0.1$).
    If there are *nu* actuators/controls, then the control has shape  `nu x 1`.
    - *impact_cost*:
    A negative reward for penalizing the Humanoid if the external contact forces are too large.
    $w_{impact} \times clamp(impact\\_cost\\_range, \\|F_{contact}\\|_2^2)$, where
    $w_{impact}$ is `impact_cost_weight` (default is $5\times10^{-7}$),
    $F_{contact}$ are the external contact forces (see `cfrc_ext` section on observation).

    The total reward returned is ***reward*** *=* *uph_cost + 1 - quad_ctrl_cost - quad_impact_cost*,
    and `info` will also contain the individual reward terms.


    ## Starting State
    All observations start in state
    (0.0, 0.0,  0.105, 1.0, 0.0  ... 0.0) with a uniform noise in the range of
    [-0.01, 0.01] added to the positional and velocity values (values in the table)
    for stochasticity. Note that the initial z coordinate is intentionally selected
    to be low, thereby indicating a laying down humanoid. The initial orientation is
    designed to make it face forward as well.


    ## Episode End
    #### Termination
    The Humanoid never terminates.

    #### Truncation
    The maximum duration of an episode is 1000 timesteps.


    ## Arguments
    `gymnasium.make` takes additional arguments such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.

    ```python
    import gymnasium as gym
    env = gym.make('HumanoidStandup-v5', impact_cost_weight=0.5e-6, ....)
    ```

    | Parameter                                    | Type      | Default          | Description                                                                                                                                                               |
    | -------------------------------------------- | --------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"humanoidstandup.xml"` | Path to a MuJoCo model                                                                                                                                             |
    | `uph_cost_weight`                            | **float** | `1`              | Weight for _uph_cost_ term (see section on reward)                                                                                                                        |
    | `ctrl_cost_weight`                           | **float** | `0.1`            | Weight for _quad_ctrl_cost_ term (see section on reward)                                                                                                                  |
    | `impact_cost_weight`                         | **float** | `0.5e-6`         | Weight for _impact_cost_ term (see section on reward)                                                                                                                     |
    | `impact_cost_range`                          | **float** | `(-np.inf, 10.0) | Clamps the _impact_cost_                                                                                                                                                  |
    | `reset_noise_scale`                          | **float** | `1e-2`           | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                            |
    | `exclude_current_positions_from_observation` | **bool**  | `True`           | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |
    | `include_cinert_in_observation`              | **bool**  | `True`           | Whether to include *cinert* elements in the observations.                                                                                                                 |
    | `include_cvel_in_observation`                | **bool**  | `True`           | Whether to include *cvel* elements in the observations.                                                                                                                   |
    | `include_qfrc_actuator_in_observation`       | **bool**  | `True`           | Whether to include *qfrc_actuator* elements in the observations.                                                                                                          |
    | `include_cfrc_ext_in_observation`            | **bool**  | `True`           | Whether to include *cfrc_ext* elements in the observations.                                                                                                               |

    ## Version History
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: This environment does not have a v3 release.
    * v2: All continuous control environments now use mujoco-py >= 1.50.
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0).
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
        xml_file="humanoidstandup.xml",
        frame_skip=5,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        uph_cost_weight=1,
        ctrl_cost_weight=0.1,
        impact_cost_weight=0.5e-6,
        impact_cost_range=(-np.inf, 10.0),
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
            uph_cost_weight,
            ctrl_cost_weight,
            impact_cost_weight,
            impact_cost_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._uph_cost_weight = uph_cost_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._impact_cost_weight = impact_cost_weight
        self._impact_cost_range = impact_cost_range
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

        obs_size = 47
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += 130 * include_cinert_in_observation
        obs_size += 78 * include_cvel_in_observation
        obs_size += 17 * include_qfrc_actuator_in_observation
        obs_size += 78 * include_cfrc_ext_in_observation

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
        self.do_simulation(action, self.frame_skip)
        pos_after = self.data.qpos[2]

        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = self._ctrl_cost_weight * np.square(self.data.ctrl).sum()

        quad_impact_cost = (
            self._impact_cost_weight * np.square(self.data.cfrc_ext).sum()
        )
        min_impact_cost, max_impact_cost = self._impact_cost_range
        quad_impact_cost = np.clip(quad_impact_cost, min_impact_cost, max_impact_cost)

        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        info = {
            "reward_linup": uph_cost,
            "reward_quadctrl": -quad_ctrl_cost,
            "reward_impact": -quad_impact_cost,
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_distance_from_origin": self.data.qpos[2] - self.init_qpos[2],
            "tendon_lenght": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
        }

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, False, False, info

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
            "z_distance_from_origin": self.data.qpos[2] - self.init_qpos[2],
            "tendon_lenght": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
        }
