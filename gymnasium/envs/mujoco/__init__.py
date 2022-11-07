from gymnasium.envs.mujoco.mujoco_env import MujocoEnv, MuJocoPyEnv  # isort:skip

# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gymnasium.envs.mujoco.ant import AntEnv
from gymnasium.envs.mujoco.half_cheetah import HalfCheetahEnv
from gymnasium.envs.mujoco.hopper import HopperEnv
from gymnasium.envs.mujoco.humanoid import HumanoidEnv
from gymnasium.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gymnasium.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gymnasium.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gymnasium.envs.mujoco.pusher import PusherEnv
from gymnasium.envs.mujoco.reacher import ReacherEnv
from gymnasium.envs.mujoco.swimmer import SwimmerEnv
from gymnasium.envs.mujoco.walker2d import Walker2dEnv
