__credits__ = ["MohammedBadra"]

from typing import Dict

import numpy as np
import math

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
# from gymnasium.spaces import Box
from gymnasium import spaces

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class A1SoccerEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    This environment simulates the A1 robot in a soccer-like scenario where the robot aims to control the ball and score goals. It integrates the physics of the A1 robot, the ball, and the dynamics of their interactions.

    Gymnasium includes the following versions of the environment:

    | Environment | Binding         | Notes                                       |
    | ----------- | --------------- | ------------------------------------------- |
    | A1SoccerEnv | `mujoco=>2.3.3` | Recommended (most features, the least bugs) |

    For more information see section "Version History".


    ## Action Space
    The action space is a `Box(-1, 1, (12,), float32)`. An action represents the torques applied at the hinge joints of the A1 robot.

    | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
    | --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | ... | (Fill in details for the A1 joints)     | ...         | ...         | ...                              | ...   | ...          |


    ## Observation Space
    The observation space primarily consists of the A1 robot's joint states and the ball's position and velocity.

    | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | ... | (Fill in details for the A1 joints and ball) | ... | ... | ...                              | ...   | ...                       |


    ## Rewards
    The total reward is designed to encourage the A1 robot to move the ball forward and potentially score a goal while minimizing control costs. The detailed reward components can be customized based on the specific problem setup.

    ## Starting State
    The initial states of the robot and the ball can be randomized to various extents, allowing for a diverse range of scenarios during training.

    ## Episode End
    The episode can terminate when a goal is scored or after a certain number of timesteps. Specific termination conditions can be adapted based on requirements.

    ## Arguments
    A1SoccerEnv provides several parameters to customize the environment, from robot dynamics to reward shaping. These can be passed during environment instantiation.

    | Parameter | Type      | Default | Description             |
    | --------- | --------- | ------- | ----------------------- |
    | ...       | ...       | ...     | ...                     |

    ```python
    import gymnasium as gym
    env = gym.make('A1Soccer', ctrl_cost_weight=0.1, ....)
    ```

    ## Version History
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
        xml_file: str = "a1_soccer_v1.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

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
        
        self.robot_joint_count = 12  # This is based on your previous information
        obs_size = (
            7          # ball qpos
            + 6        # ball qvel
            + 7        # robot root qpos
            + 6        # robot root qvel
            + 12       # robot joints qpos
            + 12       # robot joints qvel
        )

        self.observation_structure = {
            "ball_position": 3,  # or 2 if you're using 2D
            "ball_trajectory": 45, #or 30 # depending on whether it's 2D or 3D
            "goal_position": 2,
            "robot_position": 3,
            "robot_orientation": 4,
            "robot_joint_positions": self.data.qpos.size - exclude_current_positions_from_observation,
            "robot_joint_velocities": self.data.qvel.size
        }
        
        # Ball and robot indices in qpos and qvel, given their order in their xml file
        #ball indices
        #x, y, z for ball
        self.ball_position_indices = np.arange(0, 3)
        #dx, dy, dz for ball 
        self.ball_velocity_indices = np.arange(0, 3)

        #robot root indices
        #quaternion for robot root (qw, qx, qy, qz)
        self.robot_root_position_indices = np.arange(3, 7)
        #angular velocity (wx, wy, wz) for robot root 
        self.robot_root_velocity_indices = np.arange(3, 6)  

        #robot joint indices
        self.robot_joint_position_indices = np.arange(7, 7 + self.robot_joint_count)
        self.robot_joint_velocity_indices = np.arange(6, 6 + self.robot_joint_count)


        #observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        # #action spacea
        # self.action_space = spaces.Dict({
        #     "bezier_parameters": spaces.Box(low=-1, high=1, shape=(3, 5))})

    def calculate_reward(self, ball_position_before, ball_position_after, ctrl_cost):
        # Upright reward
        upright_vector = np.array([0, 0, 1])  # Assuming z-axis is up
        robot_up_vector = self.data.body_xpos[self.robot_body_index][2]  # Adjust this as needed
        upright_reward = np.dot(upright_vector, robot_up_vector)

        # Shooting towards goal reward
        goal_center = np.array([self.goal_position_x, self.goal_position_y, 0])  # Adjust as needed
        ball_movement_direction = ball_position_after - ball_position_before
        distance_to_goal_after_action = np.linalg.norm(goal_center - ball_position_after)
        distance_to_goal_before_action = np.linalg.norm(goal_center - ball_position_before)
        goal_reward = distance_to_goal_before_action - distance_to_goal_after_action  # Positive if ball moved closer to the goal

        # Combine rewards
        # self._upright_weight = 10000
        reward = self._upright_weight * upright_reward + self._goal_weight * goal_reward - ctrl_cost

        return reward
    
    # new step function
    # def step(self, action):
    #     # Ball's position before the action
    #     ball_position_before = self.data.qpos[self.ball_qpos_index]
        
    #     self.do_simulation(action, self.frame_skip)
        
    #     # Ball's position after the action
    #     ball_position_after = self.data.qpos[self.ball_qpos_index]
        
    #     ctrl_cost = self.control_cost(action)
        
    #     # Calculate reward using a separate method
    #     reward = self.calculate_reward(ball_position_before, ball_position_after, ctrl_cost)
        
    #     observation = self._get_obs()
        
    #     info = {
    #         "ball_position": ball_position_after,
    #         "ball_movement_towards_goal": ball_position_after - ball_position_before,
    #         "reward_forward": self._forward_reward_weight * (ball_position_after - ball_position_before),
    #         "reward_ctrl": -ctrl_cost,
    #     }
        
    #     if self.render_mode == "human":
    #         self.render()
        
    #     return observation, reward, False, info

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        # Flatten the observations into an array
        obs = np.concatenate([
            qpos[:7],      # ball qpos
            qvel[:6],      # ball qvel
            qpos[7:14],    # robot root qpos
            qvel[6:12],    # robot root qvel
            qpos[14:26],   # robot joints qpos
            qvel[12:24]    # robot joints qvel
        ])

        return obs
    
    def print_obs(self):
        obs_array = self._get_obs()
        
        obs_dict = {
            "ball_qpos": obs_array[:7],
            "ball_qvel": obs_array[7:13],
            "robot_root_qpos": obs_array[13:20],
            "robot_root_qvel": obs_array[20:26],
            "robot_joints_qpos": obs_array[26:38],
            "robot_joints_qvel": obs_array[38:50]
        }

        for key, value in obs_dict.items():
            print(f"{key}:\n{value}\n")

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # Initial positions and velocities
        qpos = np.array(self.init_qpos)
        qvel = np.array(self.init_qvel)

        # Modify robot's orientation to face the goal
        robot_root_orientation_indices = slice(10, 14)

        theta = math.radians(90)
        qw = math.cos(theta / 2)
        qx = 0
        qy = 0
        qz = math.sin(theta / 2)

        robot_orientation = [qw, qx, qy, qz]
        qpos[robot_root_orientation_indices] = robot_orientation

        # Introduce randomness to the ball's initial position
        ball_offset = np.random.uniform(-0.05, 0.05, size=(3,))
        qpos[:3] += ball_offset  # Assuming ball's qpos is first

        # Optional: Introduce some randomness to the robot's root position and orientation for more variability
        qpos[7:10] += self.np_random.uniform(low=noise_low, high=noise_high, size=3)  # Robot root position
        qpos[robot_root_orientation_indices] += self.np_random.uniform(low=-0.05, high=0.05, size=4)  # Robot root orientation

        # Optional: Introduce randomness to the robot's joint positions and velocities
        joint_noise = self.np_random.uniform(low=noise_low, high=noise_high, size=12)  # Assuming 12 joints
        qpos[14:26] += joint_noise

        # Adjusting this line to target the correct slice for robot's joint velocities:
        qvel[12:24] += joint_noise * 0.1  # Smaller noise for velocities


        # Set the state
        self.set_state(qpos, qvel)

        return self._get_obs()
    
    def _get_reset_info(self):
        return {
            "ball_x_position": self.data.qpos[0],
            "ball_y_position": self.data.qpos[1],
            "ball_z_position": self.data.qpos[2],
            "ball_linear_velocity": self.data.qvel[0:3],
            "ball_angular_velocity": self.data.qvel[3:6],
            
            "robot_root_x_position": self.data.qpos[7],
            "robot_root_y_position": self.data.qpos[8],
            "robot_root_z_position": self.data.qpos[9],
            "robot_root_orientation": self.data.qpos[10:14],  # Quaternion (qw, qx, qy, qz)
            "robot_root_linear_velocity": self.data.qvel[6:9],
            "robot_root_angular_velocity": self.data.qvel[9:12],

            "robot_joint_positions": self.data.qpos[14:26],
            "robot_joint_velocities": self.data.qvel[12:24]}

