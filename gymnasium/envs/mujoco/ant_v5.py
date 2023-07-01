__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Tuple, Union

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class AntEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    This environment is based on the environment introduced by Schulman,
    Moritz, Levine, Jordan and Abbeel in ["High-Dimensional Continuous Control
    Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).
    The ant is a 3D robot consisting of one torso (free rotational body) with
    four legs attached to it with each leg having two body parts. The goal is to
    coordinate the four legs to move in the forward (right) direction by applying
    torques on the eight hinges connecting the two body parts of each leg and the torso
    (nine body parts and eight hinges).


    ## Action Space
    The action space is a `Box(-1, 1, (8,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the rotor between the torso and back right hip  | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
    | 1   | Torque applied on the rotor between the back right two links      | -1          | 1           | angle_4 (right_back_leg)         | hinge | torque (N m) |
    | 2   | Torque applied on the rotor between the torso and front left hip  | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between the front left two links      | -1          | 1           | angle_1 (front_left_leg)         | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between the torso and front right hip | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between the front right two links     | -1          | 1           | angle_2 (front_right_leg)        | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the torso and back left hip   | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between the back left two links       | -1          | 1           | angle_3 (back_leg)               | hinge | torque (N m) |


    ## Observation Space
    Observations consist of positional values of different body parts of the ant,
    followed by the velocities of those individual parts (their derivatives) with all
    the positions ordered before all the velocities.

    By default, observations do not include the x- and y-coordinates of the ant's torso. These may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will be a `Box(-Inf, Inf, (107,), float64)` where the first two observations
    represent the x- and y- coordinates of the ant's torso.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x- and y-coordinates
    of the torso will be returned in `info` with keys `"x_position"` and `"y_position"`, respectively.

    However, by default, observation Space is a `Box(-Inf, Inf, (105,), float64)` where the elements correspond to the following:

    | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Unit                     |
    |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
    | 0   | z-coordinate of the torso (centre)                           | -Inf   | Inf    | torso                                  | free  | position (m)             |
    | 1   | x-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
    | 2   | y-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
    | 3   | z-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
    | 4   | w-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
    | 5   | angle between torso and first link on front left             | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
    | 6   | angle between the two links on the front left                | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
    | 7   | angle between torso and first link on front right            | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
    | 8   | angle between the two links on the front right               | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
    | 9   | angle between torso and first link on back left              | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
    | 10  | angle between the two links on the back left                 | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
    | 11  | angle between torso and first link on back right             | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
    | 12  | angle between the two links on the back right                | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
    | 13  | x-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
    | 14  | y-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
    | 15  | z-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
    | 16  | x-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
    | 17  | y-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
    | 18  | z-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
    | 19  | angular velocity of angle between torso and front left link  | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
    | 20  | angular velocity of the angle between front left links       | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
    | 21  | angular velocity of angle between torso and front right link | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
    | 22  | angular velocity of the angle between front right links      | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
    | 23  | angular velocity of angle between torso and back left link   | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
    | 24  | angular velocity of the angle between back left links        | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
    | 25  | angular velocity of angle between torso and back right link  | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
    | 26  | angular velocity of the angle between back right links       | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
    | excluded | x-coordinate of the torso (centre)                      | -Inf   | Inf    | torso                                  | free  | position (m)             |
    | excluded | y-coordinate of the torso (centre)                      | -Inf   | Inf    | torso                                  | free  | position (m)             |


    Additionally, after all the positional and velocity based values in the table, the observation contains:
    - *cfrc_ext:* 13*6 = 78 elements, which are contact forces
    (external forces - force x, y, z and torque x, y, z) applied to the
    center of mass of each of the body parts.

    The body parts are:

    | id (for `v2`, `v3`, `v4)` | id (for `v5`) | body part |
    | ---|  ---   |  ------------  |
    | 0  |excluded| worldbody (note: all values are constant 0) |
    | 1  |0       | torso |
    | 2  |1       | front_left_leg |
    | 3  |2       | aux_1 (front left leg) |
    | 4  |3       | ankle_1 (front left leg) |
    | 5  |4       | front_right_leg |
    | 6  |5       | aux_2 (front right leg) |
    | 7  |6       | ankle_2 (front right leg) |
    | 8  |7       | back_leg (back left leg) |
    | 9  |8       | aux_3 (back left leg) |
    | 10 |9       | ankle_3 (back left leg) |
    | 11 |10      | right_back_leg |
    | 12 |11      | aux_4 (back right leg) |
    | 13 |12      | ankle_4 (back right leg) |

    The (x,y,z) coordinates are translational DOFs while the orientations are rotational
    DOFs expressed as quaternions. One can read more about free joints in the [Mujoco Documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).


    **Note:** Ant-v4+ environment no longer has the following contact forces issue.
    If using previous Humanoid versions from v4, there have been reported issues that using a Mujoco-Py version > 2.0 results
    in the contact forces always being 0. As such we recommend to use a Mujoco-Py version < 2.0
    when using the Ant environment if you would like to report results with contact forces (if
    contact forces are not used in your experiments, you can use version > 2.0).


    ## Rewards
    The reward consists of four parts:
    - *healthy_reward*:
    Every timestep that the Ant is healthy (see definition in section "Episode Termination"),
    it gets a reward of fixed value `healthy_reward`.
    - *forward_reward*:
    A reward for moving forward,
    this reward would be positive if the Ant moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the `main_body` ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which is depends on the `frame_skip` parameter (default is 5),
    and `frametime`, which is 0.01 - so the default is $dt = 5 \times 0.01 = 0.05$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1$).
    - *ctrl_cost*:
    A negative reward to penalize the Ant for taking actions that are too large.
    $w_{control} \times \\|action\\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $0.5$).
    - *contact_cost*:
    A negative reward to penalize the Ant if the external contact forces are too large.
    $w_{contact} \times \\|F_{contact}\\|_2^2$, where
    $w_{contact}$ is `contact_cost_weight` (default is $5\times10^{-4}$),
    $F_{contact}$ are the external contact forces clipped by `contact_force_range` (see `cfrc_ext` section on observation).

    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost*.
    and `info` will also contain the individual reward terms.

    But if `use_contact_forces=false` on `v4`
    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*.


    ## Starting State
    All observations start in state
    (0.0, 0.0,  0.75, 1.0, 0.0  ... 0.0) with a uniform noise in the range
    of [-`reset_noise_scale`, `reset_noise_scale`] added to the positional values and standard normal noise
    with mean 0 and standard deviation `reset_noise_scale` added to the velocity values for
    stochasticity. Note that the initial z coordinate is intentionally selected
    to be slightly high, thereby indicating a standing up ant. The initial orientation
    is designed to make it face forward as well.


    ## Episode End
    #### Termination
    If `terminate_when_unhealthy is True` (which is the default), the environment terminates when the Ant is unhealthy.
    the Ant is unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite
    2. The z-coordinate of the torso is **not** in the closed interval given by `healthy_z_range` (defaults to [0.2, 1.0])

    #### truncation
    The maximum duration of an episode is 1000 timesteps.


    ## Arguments
    `gymnasium.make` takes additional arguments such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.

    ```python
    import gymnasium as gym
    env = gym.make('Ant-v5', ctrl_cost_weight=0.5, ...)
    ```

    | Parameter                                  | Type       | Default      |Description                    |
    |--------------------------------------------|------------|--------------|-------------------------------|
    |`xml_file`                                  | **str**    | `"ant.xml"`  | Path to a MuJoCo model |
    |`forward_reward_weight`                     | **float**  | `1`          | Weight for _forward_reward_ term (see section on reward)|
    |`ctrl_cost_weight`                          | **float**  | `0.5`        | Weight for _ctrl_cost_ term (see section on reward) |
    |`contact_cost_weight`                       | **float**  | `5e-4`       | Weight for _contact_cost_ term (see section on reward) |
    |`healthy_reward`                            | **float**  | `1`          | Weight for _healthy_reward_ term (see section on reward) |
    |`main_body`                                 |**str|int** | `1`("torso") | Name or ID of the body, whose displacement is used to calculate the *dx*/_forward_reward_ (useful for custom MuJoCo models)|
    |`terminate_when_unhealthy`                  | **bool**   | `True`       | If true, issue a done signal if the z-coordinate of the torso is no longer in the `healthy_z_range` |
    |`healthy_z_range`                           | **tuple**  | `(0.2, 1)`   | The ant is considered healthy if the z-coordinate of the torso is in this range |
    |`contact_force_range`                       | **tuple**  | `(-1, 1)`    | Contact forces are clipped to this range in the computation of *contact_cost* |
    |`reset_noise_scale`                         | **float**  | `0.1`        | Scale of random perturbations of initial position and velocity (see section on Starting State) |
    |`exclude_current_positions_from_observation`| **bool**   | `True`       | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |
    |`include_cfrc_ext_in_observation`           | **bool**   | `True`       | Whether to include *cfrc_ext* elements in the observations. |
    |`use_contact_forces` (`v4` only)            | **bool**   | `False`      | If true, it extends the observation space by adding contact forces (see `Observation Space` section) and includes contact_cost to the reward function (see `Rewards` section) |

    ## Version History
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3, also removed contact forces from the default observation space (new variable `use_contact_forces=True` can restore them)
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
        xml_file: str = "ant.xml",
        frame_skip: int = 5,
        default_camera_config: Dict = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.2, 1.0),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
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
            healthy_reward,
            main_body,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
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
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        # TODO remove after validation
        assert terminated == (
            not self.is_healthy if self._terminate_when_unhealthy else False
        )
        return terminated

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        # TODO remove after validation
        # assert (xy_position_before == self.get_body_com("torso")[:2].copy()).all()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()
        # TODO remove after validation
        # assert (xy_position_after == self.get_body_com("torso")[:2].copy()).all()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._include_cfrc_ext_in_observation:
            assert (
                self.contact_forces[0].flat.copy() == 0
            ).all()  # TODO remove after validation
            contact_force = self.contact_forces[1:].flat.copy()
            assert len(contact_force) == 78
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
