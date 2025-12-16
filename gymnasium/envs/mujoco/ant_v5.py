__credits__ = ["Kallinteris-Andreas"]

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
    This environment is based on the one introduced by Schulman, Moritz, Levine, Jordan, and Abbeel in ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).
    The ant is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts.
    The goal is to coordinate the four legs to move in the forward (right) direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges).

    Note: Although the robot is called "Ant", it is actually 75cm tall and weighs 910.88g, with the torso being 327.25g and each leg being 145.91g.

    ## Action Space
    ```{figure} action_space_figures/ant.png
    :name: ant
    ```

    The action space is a `Box(-1, 1, (8,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
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
    The observation space consists of the following parts (in order):

    - *qpos (13 elements by default):* Position values of the robot's body parts.
    - *qvel (14 elements):* The velocities of these individual body parts (their derivatives).
    - *cfrc_ext (78 elements):* This is the center of mass based external forces on the body parts.
    It has shape 13 * 6 (*nbody * 6*) and hence adds another 78 elements to the state space.
    (external forces - force x, y, z and torque x, y, z)

    By default, the observation does not include the x- and y-coordinates of the torso.
    These can be included by passing `exclude_current_positions_from_observation=False` during construction.
    In this case, the observation space will be a `Box(-Inf, Inf, (107,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
    Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

    By default, however, the observation space is a `Box(-Inf, Inf, (105,), float64)`, where the position and velocity elements are as follows:

    | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Type (Unit)              |
    |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
    | 0   | z-coordinate of the torso (centre)                           | -Inf   | Inf    | root                                   | free  | position (m)             |
    | 1   | w-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 2   | x-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 3   | y-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 4   | z-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 5   | angle between torso and first link on front left             | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
    | 6   | angle between the two links on the front left                | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
    | 7   | angle between torso and first link on front right            | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
    | 8   | angle between the two links on the front right               | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
    | 9   | angle between torso and first link on back left              | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
    | 10  | angle between the two links on the back left                 | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
    | 11  | angle between torso and first link on back right             | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
    | 12  | angle between the two links on the back right                | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
    | 13  | x-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
    | 14  | y-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
    | 15  | z-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
    | 16  | x-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
    | 17  | y-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
    | 18  | z-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
    | 19  | angular velocity of angle between torso and front left link  | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
    | 20  | angular velocity of the angle between front left links       | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
    | 21  | angular velocity of angle between torso and front right link | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
    | 22  | angular velocity of the angle between front right links      | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
    | 23  | angular velocity of angle between torso and back left link   | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
    | 24  | angular velocity of the angle between back left links        | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
    | 25  | angular velocity of angle between torso and back right link  | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
    | 26  | angular velocity of the angle between back right links       | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
    | excluded | x-coordinate of the torso (centre)                      | -Inf   | Inf    | root                                   | free  | position (m)             |
    | excluded | y-coordinate of the torso (centre)                      | -Inf   | Inf    | root                                   | free  | position (m)             |

    The body parts are:

    | body part                 | id (for `v2`, `v3`, `v4)` | id (for `v5`) |
    |  -----------------------  |  ---   |  ---  |
    | worldbody (note: all values are constant 0) | 0  |excluded|
    | torso                     | 1  |0       |
    | front_left_leg            | 2  |1       |
    | aux_1 (front left leg)    | 3  |2       |
    | ankle_1 (front left leg)  | 4  |3       |
    | front_right_leg           | 5  |4       |
    | aux_2 (front right leg)   | 6  |5       |
    | ankle_2 (front right leg) | 7  |6       |
    | back_leg (back left leg)  | 8  |7       |
    | aux_3 (back left leg)     | 9  |8       |
    | ankle_3 (back left leg)   | 10 |9       |
    | right_back_leg            | 11 |10      |
    | aux_4 (back right leg)    | 12 |11      |
    | ankle_4 (back right leg)  | 13 |12      |

    The (x,y,z) coordinates are translational DOFs, while the orientations are rotational DOFs expressed as quaternions.
    One can read more about free joints in the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).


    **Note:**
    When using Ant-v3 or earlier versions, problems have been reported when using a `mujoco-py` version > 2.0, resulting in  contact forces always being 0.
    Therefore, it is recommended to use a `mujoco-py` version < 2.0 when using the Ant environment if you want to report results with contact forces (if contact forces are not used in your experiments, you can use version > 2.0).


    ## Rewards
    The total reward is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost*.

    - *healthy_reward*:
    Every timestep that the Ant is healthy (see definition in section "Episode End"),
    it gets a reward of fixed value `healthy_reward` (default is $1$).
    - *forward_reward*:
    A reward for moving forward,
    this reward would be positive if the Ant moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the `main_body` ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
    and `frametime`, which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1$).
    - *ctrl_cost*:
    A negative reward to penalize the Ant for taking actions that are too large.
    $w_{control} \times \|action\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $0.5$).
    - *contact_cost*:
    A negative reward to penalize the Ant if the external contact forces are too large.
    $w_{contact} \times \|F_{contact}\|_2^2$, where
    $w_{contact}$ is `contact_cost_weight` (default is $5\times10^{-4}$),
    $F_{contact}$ are the external contact forces clipped by `contact_force_range` (see `cfrc_ext` section on Observation Space).

    `info` contains the individual reward terms.

    But if `use_contact_forces=False` on `v4`
    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*.


    ## Starting State
    The initial position state is $[0.0, 0.0, 0.75, 1.0, 0.0, ... 0.0] + \mathcal{U}_{[-reset\_noise\_scale \times I_{15}, reset\_noise\_scale \times I_{15}]}$.
    The initial velocity state is $\mathcal{N}(0_{14}, reset\_noise\_scale^2 \times I_{14})$.

    where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.

    Note that the z- and x-coordinates are non-zero so that the ant can immediately stand up and face forward (x-axis).


    ## Episode End
    ### Termination
    If `terminate_when_unhealthy is True` (the default), the environment terminates when the Ant is unhealthy.
    the Ant is unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite.
    2. The z-coordinate of the torso (the height) is **not** in the closed interval given by the `healthy_z_range` argument (default is $[0.2, 1.0]$).

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    Ant provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('Ant-v5', ctrl_cost_weight=0.5, ...)
    ```

    | Parameter                                  | Type       | Default      |Description                    |
    |--------------------------------------------|------------|--------------|-------------------------------|
    |`xml_file`                                  | **str**    | `"ant.xml"`  | Path to a MuJoCo model                                                                                                                                                                                      |
    |`forward_reward_weight`                     | **float**  | `1`          | Weight for _forward_reward_ term (see `Rewards` section)                                                                                                                                                    |
    |`ctrl_cost_weight`                          | **float**  | `0.5`        | Weight for _ctrl_cost_ term (see `Rewards` section)                                                                                                                                                         |
    |`contact_cost_weight`                       | **float**  | `5e-4`       | Weight for _contact_cost_ term (see `Rewards` section)                                                                                                                                                      |
    |`healthy_reward`                            | **float**  | `1`          | Weight for _healthy_reward_ term (see `Rewards` section)                                                                                                                                                    |
    |`main_body`                                 |**str\|int**| `1`("torso") | Name or ID of the body, whose displacement is used to calculate the *dx*/_forward_reward_ (useful for custom MuJoCo models) (see `Rewards` section)                                                         |
    |`terminate_when_unhealthy`                  | **bool**   | `True`       | If `True`, issue a `terminated` signal is unhealthy (see `Episode End` section)                                                                                                                                |
    |`healthy_z_range`                           | **tuple**  | `(0.2, 1)`   | The ant is considered healthy if the z-coordinate of the torso is in this range (see `Episode End` section)                                                                                                 |
    |`contact_force_range`                       | **tuple**  | `(-1, 1)`    | Contact forces are clipped to this range in the computation of *contact_cost* (see `Rewards` section)                                                                                                       |
    |`reset_noise_scale`                         | **float**  | `0.1`        | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
    |`exclude_current_positions_from_observation`| **bool**   | `True`       | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |
    |`include_cfrc_ext_in_observation`           | **bool**   | `True`       | Whether to include *cfrc_ext* elements in the observations (see `Observation State` section)                                                                                                                |
    |`use_contact_forces` (`v4` only)            | **bool**   | `False`      | If `True`, it extends the observation space by adding contact forces (see `Observation Space` section) and includes contact_cost to the reward function (see `Rewards` section)                             |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
        - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Fixed bug: `healthy_reward` was given on every step (even if the Ant is unhealthy), now it is only given when the Ant is healthy. The `info["reward_survive"]` is updated with this change (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/526)).
        - The reward function now always includes `contact_cost`, before it was only included if `use_contact_forces=True` (can be set to `0` with `contact_cost_weight=0`).
        - Excluded the `cfrc_ext` of `worldbody` from the observation space, as it was always 0 and thus provided no useful information to the agent, resulting in slightly faster training (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/204)).
        - Added the `main_body` argument, which specifies the body used to compute the forward reward (mainly useful for custom MuJoCo models).
        - Added the `forward_reward_weight` argument, which defaults to `1` (effectively the same behavior as in `v4`).
        - Added the `include_cfrc_ext_in_observation` argument, previously in `v4` the inclusion of `cfrc_ext` observations was controlled by `use_contact_forces` which defaulted to `False`, while `include_cfrc_ext_in_observation` defaults to `True`.
        - Removed the `use_contact_forces` argument (note: its functionality has been replaced by `include_cfrc_ext_in_observation` and `contact_cost_weight`) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/214)).
        - Fixed `info["reward_ctrl"]` sometimes containing `contact_cost` instead of `ctrl_cost`.
        - Fixed `info["x_position"]` & `info["y_position"]` & `info["distance_from_origin"]` giving `xpos` instead of `qpos` observations (`xpos` observations are behind 1 `mj_step()` more [here](https://github.com/deepmind/mujoco/issues/889#issuecomment-1568896388)) (related [GitHub issue #1](https://github.com/Farama-Foundation/Gymnasium/issues/521) & [GitHub issue #2](https://github.com/Farama-Foundation/Gymnasium/issues/539)).
        - Removed `info["forward_reward"]` as it is equivalent to `info["reward_forward"]`.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3, also removed contact forces from the default observation space (new variable `use_contact_forces=True` can restore them).
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen). Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
    * v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "ant.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        main_body: int | str = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.2, 1.0),
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
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
                "rgbd_tuple",
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

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
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
