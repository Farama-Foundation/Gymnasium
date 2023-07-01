__credits__ = ["Kallinteris-Andreas"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}


class InvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment is the cartpole environment based on the work done by
    Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can
    solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    just like in the classic environments but now powered by the Mujoco physics simulator -
    allowing for more complex experiments (such as varying the effects of gravity).
    This environment involves a cart that can moved linearly, with a pole fixed on it
    at one end and having another end free. The cart can be pushed left or right, and the
    goal is to balance the pole on the top of the cart by applying forces on the cart.


    ## Action Space
    The agent take a 1-element vector for actions.

    The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
    the numerical force applied to the cart (with magnitude representing the amount of
    force and sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |


    ## Observation Space
    The state space consists of positional values of different body parts of
    the pendulum system, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(4,)` where the elements correspond to the following:

    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Unit                      |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
    | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
    | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
    | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | anglular velocity (rad/s) |


    ## Rewards
    The goal is to keep the inverted pendulum stand upright (within a certain angle limit)
    for as long as possible - as such a reward of +1 is awarded for each timestep that
    the pole is upright.

    The pole is considered upright if:
    $|angle| < 0.2$.

    and `info` also contains the reward.


    ## Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
    of `[-reset_noise_scale, reset_noise_scale]` added to the values for stochasticity.


    ## Episode End
    #### Termination
    The environment terminates when the Inverted Pendulum is unhealthy.
    The Inverted Pendulum is unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite.
    2. The absolute value of the vertical angle between the pole and the cart is greater than 0.2 radian.

    #### Truncation
    The maximum duration of an episode is 1000 timesteps.


    ## Arguments
    `gymnasium.make` takes additional arguments such as `reset_noise_scale`.

    ```python
    import gymnasium as gym
    env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
    ```

    | Parameter               | Type       | Default      |Description                    |
    |-------------------------|------------|--------------|-------------------------------|
    | `xml_file`              | **str**    | `"inverted_double_pendulum.xml"`  | Path to a MuJoCo model |
    | `reset_noise_scale`     | **float**  | `0.01`        | Scale of random perturbations of initial position and velocity (see section on Starting State) |

    ## Version History
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1..
    * v3: This environment does not have a v3 release.
    * v2: All continuous control environments now use mujoco-py >= 1.5.
    * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum).
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
        xml_file="inverted_pendulum.xml",
        frame_skip=2,
        reset_noise_scale=0.01,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
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

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        terminated = bool(
            not np.isfinite(observation).all() or (np.abs(observation[1]) > 0.2)
        )

        reward = int(not terminated)

        info = {"reward_survive": reward}

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()
