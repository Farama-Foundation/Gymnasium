__credits__ = ["Kallinteris-Andreas"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}


class InvertedDoublePendulumEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    This environment originates from control theory and builds on the cartpole
    environment based on the work done by Barto, Sutton, and Anderson in
    ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    powered by the Mujoco physics simulator - allowing for more complex experiments
    (such as varying the effects of gravity or constraints). This environment involves a cart that can
    moved linearly, with a pole fixed on it and a second pole fixed on the other end of the first one
    (leaving the second pole as the only one with one free end). The cart can be pushed left or right,
    and the goal is to balance the second pole on top of the first pole, which is in turn on top of the
    cart, by applying continuous forces on the cart.

    Gymnasium includes the following versions of the environment:

    | Environment               | Binding         | Notes                                       |
    | ------------------------- | --------------- | ------------------------------------------- |
    | InvertedDoublePendulum-v5 | `mujoco=>2.3.3` | Recommended (most features, the least bugs) |
    | InvertedDoublePendulum-v4 | `mujoco=>2.1.3` | Maintained for reproducibility              |
    | InvertedDoublePendulum-v2 | `mujoco-py`     | Maintained for reproducibility              |

    For more information see section "Version History".

    ## Action Space
    The agent take a 1-element vector for actions.
    The action space is a continuous `(action)` in `[-1, 1]`, where `action` represents the
    numerical force applied to the cart (with magnitude representing the amount of force and
    sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint |Type (Unit)|
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -1          | 1           | slider                           | slide | Force (N) |


    ## Observation Space
    The state space consists of positional values of different body parts of the pendulum system,
    followed by the velocities of those individual parts (their derivatives) with all the
    positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(9,)` where the elements correspond to the following:

    | Num | Observation                                                       | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | ----------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | position of the cart along the linear surface                     | -Inf | Inf | slider                           | slide | position (m)             |
    | 1   | sine of the angle between the cart and the first pole             | -Inf | Inf | sin(hinge)                       | hinge | unitless                 |
    | 2   | sine of the angle between the two poles                           | -Inf | Inf | sin(hinge2)                      | hinge | unitless                 |
    | 3   | cosine of the angle between the cart and the first pole           | -Inf | Inf | cos(hinge)                       | hinge | unitless                 |
    | 4   | cosine of the angle between the two poles                         | -Inf | Inf | cos(hinge2)                      | hinge | unitless                 |
    | 5   | velocity of the cart                                              | -Inf | Inf | slider                           | slide | velocity (m/s)           |
    | 6   | angular velocity of the angle between the cart and the first pole | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the angle between the two poles               | -Inf | Inf | hinge2                           | hinge | angular velocity (rad/s) |
    | 8   | constraint force - x                                              | -Inf | Inf | slider                           | slide | Force (N)                |
    | excluded | constraint force - y                                         | -Inf | Inf | hinge                            | slide | Force (N)                |
    | excluded | constraint force - z                                         | -Inf | Inf | hinge2                           | slide | Force (N)                |


    There is physical contact between the robots and their environment - and Mujoco
    attempts at getting realistic physics simulations for the possible physical contact
    dynamics by aiming for physical accuracy and computational efficiency.

    There is one constraint force for contacts for each degree of freedom (3).
    The approach and handling of constraints by Mujoco is unique to the simulator and is based on their research.
    Once can find more information in their [*documentation*](https://mujoco.readthedocs.io/en/latest/computation.html)
    or in their paper ["Analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo"](https://homes.cs.washington.edu/~todorov/papers/TodorovICRA14.pdf).


    ## Rewards
    The total reward is: ***reward*** *=* *alive_bonus - distance_penalty - velocity_penalty*.

    - *alive_bonus*:
    The goal is to keep the second inverted pendulum upright (within a certain angle limit) as long as possible -
    so for each timestep that the second pole is upright, a reward of `healthy_reward` is given.
    - *distance_penalty*:
    This reward is a measure of how far the *tip* of the second pendulum (the only free end) moves,
    and it is calculated as $0.01 x_{pole2-tip}^2 + (y_{pole2-tip}-2)^2$,
    where $x_{pole2-tip}, y_{pole2-tip}$ are the xy-coordinatesof the tip of the second pole.
    - *velocity_penalty*:
    A negative reward to penalize the agent for moving too fast.
    $10^{-3} \omega_1 + 5 10^{-3} \omega_2$,
    where $\omega_1, \omega_2$ are the angular velocities of the hinges.

    `info` contains the individual reward terms.


    ## Starting State
    The initial position state is $\mathcal{U}_{[-reset\_noise\_scale \times 1_{3}, reset\_noise\_scale \times 1_{3}]}$.
    The initial velocity state is $\mathcal{N}(0_{3}, reset\_noise\_scale^2 \times I_{3})$.

    where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.


    ## Episode End
    #### Termination
    The environment terminates when the Inverted Double Pendulum is unhealthy.
    The Inverted Double Pendulum is unhealthy if any of the following happens:

    1.Termination: The y_coordinate of the tip of the second pole $\leq 1$.

    Note: The maximum standing height of the system is 1.2 m when all the parts are perpendicularly vertical on top of each other.

    #### Truncation
    The default duration of an episode is 1000 timesteps


    ## Arguments
    InvertedDoublePendulum provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('InvertedDoublePendulum-v5', healthy_reward=10, ...)
    ```

    | Parameter               | Type       | Default                        | Description                   |
    |-------------------------|------------|--------------                  |-------------------------------|
    | `xml_file`              | **str**    |`"inverted_double_pendulum.xml"`| Path to a MuJoCo model |
    | `healthy_reward`        | **float**  | `10                            | Constant reward given if the pendulum is "healthy" (upright) |
    | `reset_noise_scale`     | **float**  | `0.1`                          | Scale of random perturbations of initial position and velocity (see section on Starting State) |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Fixed bug: `healthy_reward` was given on every step (even if the Pendulum is unhealthy), now it is only given if the DoublePendulum is healthy (not terminated)(related [Github issue](https://github.com/Farama-Foundation/Gymnasium/issues/500)).
        - Excluded the `qfrc_constraint` ("constraint force") of the hinges from the observation space (as it was always 0, thus providing no useful information to the agent, resulting is slightly faster training) (related [Github issue](https://github.com/Farama-Foundation/Gymnasium/issues/228)).
        - Added `xml_file` argument.
        - Added `reset_noise_scale` argument, to set the range of initial states.
        - Added `healthy_reward` argument to configure the reward function (defaults are effectively the same as in `v4`).
        - Added individual reward terms in `info` ( `info["reward_survive"]`, `info["distance_penalty"]`, `info["velocity_penalty"]`).
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v3: This environment does not have a v3 release.
    * v2: All continuous control environments now use mujoco-py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum)
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
        xml_file: str = "inverted_double_pendulum.xml",
        frame_skip: int = 5,
        healthy_reward: float = 10.0,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)

        self._healthy_reward = healthy_reward
        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)

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

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        x, _, y = self.data.site_xpos[0]
        v1, v2 = self.data.qvel[1:3]

        terminated = bool(y <= 1)

        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = self._healthy_reward * int(not terminated)
        reward = alive_bonus - dist_penalty - vel_penalty

        info = {
            "reward_survive": alive_bonus,
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                np.sin(self.data.qpos[1:]),  # link angles
                np.cos(self.data.qpos[1:]),
                np.clip(self.data.qvel, -10, 10),
                np.clip(self.data.qfrc_constraint, -10, 10)[:1],
            ]
        ).ravel()

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.set_state(
            self.init_qpos
            + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nq
            ),
            self.init_qvel
            + self.np_random.standard_normal(self.model.nv) * self._reset_noise_scale,
        )
        return self._get_obs()
