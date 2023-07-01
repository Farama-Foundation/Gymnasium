__credits__ = ["Kallinteris-Andreas"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class ReacherEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    "Reacher" is a two-jointed robot arm. The goal is to move the robot's end effector (called *fingertip*) close to a
    target that is spawned at a random position.


    ## Action Space
    The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.

    | Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
    |-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
    | 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1 | 1 | joint0  | hinge | torque (N m) |
    | 1   |  Torque applied at the second hinge (connecting the two links)                  | -1 | 1 | joint1  | hinge | torque (N m) |


    ## Observation Space
    Observations consist of

    - The cosine of the angles of the two arms
    - The sine of the angles of the two arms
    - The coordinates of the target
    - The angular velocities of the arms
    - The vector between the target and the reacher's fingertip (3 dimensional with the last element being 0)

    The observation is a `Box(-Inf, Inf, (10,), float64)` where the elements correspond to the following:

    | Num | Observation                                                                                    | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ---------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | cosine of the angle of the first arm                                                           | -Inf | Inf | cos(joint0)                      | hinge | unitless                 |
    | 1   | cosine of the angle of the second arm                                                          | -Inf | Inf | cos(joint1)                      | hinge | unitless                 |
    | 2   | sine of the angle of the first arm                                                             | -Inf | Inf | sin(joint0)                      | hinge | unitless                 |
    | 3   | sine of the angle of the second arm                                                            | -Inf | Inf | sin(joint1)                      | hinge | unitless                 |
    | 4   | x-coordinate of the target                                                                     | -Inf | Inf | target_x                         | slide | position (m)             |
    | 5   | y-coordinate of the target                                                                     | -Inf | Inf | target_y                         | slide | position (m)             |
    | 6   | angular velocity of the first arm                                                              | -Inf | Inf | joint0                           | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the second arm                                                             | -Inf | Inf | joint1                           | hinge | angular velocity (rad/s) |
    | 8   | x-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
    | 9   | y-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
    | excluded | z-value of position_fingertip - position_target (constantly 0 since reacher is 2d)        | -Inf | Inf | NA                               | slide | position (m)             |

    Most Gym environments just return the positions and velocity of the
    joints in the `.xml` file as the state of the environment. However, in
    reacher the state is created by combining only certain elements of the
    position and velocity, and performing some function transformations on them.
    If one is to read the `reacher.xml` then they will find 4 joints:

    | Num | Observation                 | Min      | Max      | Name (in corresponding XML file) | Joint | Unit               |
    |-----|-----------------------------|----------|----------|----------------------------------|-------|--------------------|
    | 0   | angle of the first arm      | -Inf     | Inf      | joint0                           | hinge | angle (rad)        |
    | 1   | angle of the second arm     | -Inf     | Inf      | joint1                           | hinge | angle (rad)        |
    | 2   | x-coordinate of the target  | -Inf     | Inf      | target_x                         | slide | position (m)       |
    | 3   | y-coordinate of the target  | -Inf     | Inf      | target_y                         | slide | position (m)       |


    ## Rewards
    The reward consists of two parts:
    - *reward_distance*:
    This reward is a measure of how far the *fingertip* of the reacher (the unattached end) is from the target,
    with a more negative value assigned for if the reacher's *fingertip* is further away from the target.
    It is $-w_{near} \|(P_{fingertip} - P_{target})\|_2$.
    where $w_{near}$ is the `reward_near_weight`.
    - *reward_control*:
    A negative reward to penalize the walker for taking actions that are too large.
    It is measured as the negative squared Euclidean norm of the action, i.e. as $-w_{control} \|action\|_2^2$.
    where $w_{control}$ is the `reward_control_weight`.

    The total reward returned is ***reward*** *=* *reward_distance + reward_control*.
    `info` will also contain the individual reward terms.

    ## Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with a noise added for stochasticity. A uniform noise in the range
    [-0.1, 0.1] is added to the positional attributes, while the target position
    is selected uniformly at random in a disk of radius 0.2 around the origin.
    Independent, uniform noise in the
    range of [-0.005, 0.005] is added to the velocities, and the last
    element ("fingertip" - "target") is calculated at the end once everything
    is set. The default setting has a framerate of 2 and a *dt = 2 * 0.01 = 0.02*

    ## Episode End
    #### Termination
    The Reacher never terminates.

    #### Truncation
    The maximum duration of an episode is 50 timesteps.


    ## Arguments
    `gymnasium.make` takes additional arguments such as `xml_file`.

    ```python
    import gymnasium as gym
    env = gym.make('Reacher-v5', xml_file=...)
    ```

    | Parameter               | Type       | Default      |Description                                               |
    |-------------------------|------------|--------------|----------------------------------------------------------|
    | `xml_file`              | **str**    |`"reacher.xml"`| Path to a MuJoCo model                                  |
    | `reward_dist_weight`    | **float**  | `1`          | Weight for *reward_dist* term (see section on reward)    |
    | `reward_control_weight` | **float**  | `0.1`        | Weight for *reward_control* term (see section on reward) |

    ## Version History
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v3: This environment does not have a v3 release.
    * v2: All continuous control environments now use mujoco-py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (not including reacher, which has a max_time_steps of 50). Added reward_threshold to environments.
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
        xml_file="reacher.xml",
        frame_skip=2,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        reward_dist_weight=1,
        reward_control_weight=1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_control_weight,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight

        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
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

    def step(self, action):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec) * self._reward_dist_weight
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight
        reward = reward_dist + reward_ctrl

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        assert (self.get_body_com("fingertip") - self.get_body_com("target"))[
            2
        ] == 0  # TODO remove after validation
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qpos.flat[2:],
                self.data.qvel.flat[:2],
                (self.get_body_com("fingertip") - self.get_body_com("target"))[:2],
            ]
        )
