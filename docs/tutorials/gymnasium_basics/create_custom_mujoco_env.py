# fmt: off
"""
Create a Custom MuJoCo Environment
===================================

This tutorial demonstrates how to create a custom MuJoCo-based environment
by subclassing ``MujocoEnv``. You will learn how to define custom
observations, rewards, and termination conditions for a simulated
robotic task.

We build a simple **ReachTarget** environment: a robot arm must move its
end-effector to a randomly placed target. This is a common benchmark in
robotic manipulation and a good starting template for more complex tasks.

Prerequisites:

- Familiarity with the Gymnasium API (``Env``, ``spaces``, ``step``, ``reset``)
- Basic understanding of MuJoCo (MJCF XML format, bodies, joints, actuators)
- ``gymnasium>=1.0.0`` with MuJoCo installed (``pip install "gymnasium[mujoco]"``)
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv

# %%
# Step 1 — Define the MJCF Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# MuJoCo environments are defined by an XML file in MJCF format.
# Here we create a minimal 2-link planar arm with a target site.
# In a real project, you would save this to a ``.xml`` file.

REACHER_XML = """
<mujoco model="reach_target">
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>

    <!-- 2-link arm mounted on a fixed base -->
    <body name="link0" pos="0 0 0.5">
      <joint name="joint0" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom name="link0_geom" type="capsule" fromto="0 0 0 0.3 0 0"
            size="0.02" rgba="0.2 0.4 0.8 1"/>
      <body name="link1" pos="0.3 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0.3 0 0"
              size="0.02" rgba="0.8 0.2 0.2 1"/>
        <!-- End-effector site (used to track fingertip position) -->
        <site name="fingertip" pos="0.3 0 0" size="0.02" rgba="1 1 0 1"/>
      </body>
    </body>

    <!-- Target (visual only — no collision) -->
    <body name="target" pos="0.3 0.3 0.5" mocap="true">
      <geom name="target_geom" type="sphere" size="0.03"
            rgba="0 1 0 0.6" contype="0" conaffinity="0"/>
      <site name="target_site" pos="0 0 0" size="0.03"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor0" joint="joint0" gear="1" ctrlrange="-1 1"/>
    <motor name="motor1" joint="joint1" gear="1" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# %%
# Step 2 — Subclass ``MujocoEnv``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The key methods to implement are:
#
# - ``__init__``: set up observation and action spaces
# - ``step``: compute reward, check termination, return observation
# - ``reset_model``: randomize the environment at the start of each episode
# - ``_get_obs``: build the observation vector
#
# ``MujocoEnv`` handles MuJoCo simulation, rendering, and time-step
# management for you.


class ReachTargetEnv(MujocoEnv, utils.EzPickle):
    """A 2-link arm must reach a randomly placed target.

    Observation (8-dim):
        [cos(q0), sin(q0), cos(q1), sin(q1), dq0, dq1, dx_target, dy_target]

    Action (2-dim):
        Torques applied to the two hinge joints, clipped to [-1, 1].

    Reward:
        Negative Euclidean distance from fingertip to target,
        plus a small control penalty.

    Termination:
        The episode ends when the fingertip is within 0.02 of the target.
        The episode is truncated after ``max_episode_steps`` (default 200).
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(self, render_mode: str | None = None, **kwargs):
        # Define the observation space before calling super().__init__.
        # 8 dimensions: 4 for joint angles (cos/sin), 2 for velocities,
        # 2 for vector from fingertip to target.
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
        )

        # Write the XML to a temporary file for MujocoEnv to load.
        import tempfile, os

        self._xml_path = os.path.join(tempfile.gettempdir(), "reach_target.xml")
        with open(self._xml_path, "w") as f:
            f.write(REACHER_XML)

        # frame_skip: number of MuJoCo simulation steps per Gymnasium step.
        # Higher values = faster simulation but coarser control.
        super().__init__(
            model_path=self._xml_path,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs,
        )
        utils.EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        # Cache site and body IDs for efficient lookup.
        self._fingertip_sid = self.model.site("fingertip").id
        self._target_sid = self.model.site("target_site").id
        self._target_bid = self.model.body("target").id

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(self, action):
        # 1. Run the simulation forward.
        self.do_simulation(action, self.frame_skip)

        # 2. Compute observation.
        obs = self._get_obs()

        # 3. Compute reward: distance + control cost.
        fingertip = self.data.site_xpos[self._fingertip_sid][:2]
        target = self.data.site_xpos[self._target_sid][:2]
        distance = np.linalg.norm(fingertip - target)
        control_cost = 0.01 * np.sum(np.square(action))
        reward = -distance - control_cost

        # 4. Check termination (reached the target).
        terminated = bool(distance < 0.02)

        # 5. Info dict can carry diagnostics.
        info = {"distance": distance, "reward_ctrl": -control_cost}

        return obs, reward, terminated, False, info

    def reset_model(self):
        """Randomize joint positions and target location."""
        # Random initial joint angles (small range for reproducibility).
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        # Randomize target position within workspace.
        target_xy = self.np_random.uniform(low=-0.4, high=0.4, size=2)
        # Move mocap body to new target position.
        self.data.mocap_pos[0][:2] = target_xy

        return self._get_obs()

    def _get_obs(self):
        """Build the observation vector."""
        q = self.data.qpos.flat[:]
        dq = self.data.qvel.flat[:]

        # Fingertip-to-target vector (2D).
        fingertip = self.data.site_xpos[self._fingertip_sid][:2]
        target = self.data.site_xpos[self._target_sid][:2]
        delta = target - fingertip

        return np.concatenate(
            [
                np.cos(q),  # cos of joint angles
                np.sin(q),  # sin of joint angles
                dq,  # joint velocities
                delta,  # vector to target
            ]
        )


# %%
# Step 3 — Register and Test the Environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Gymnasium discovers environments through ``register()``.
# After registration, you can create the environment with ``gym.make()``.

import gymnasium as gym

gym.register(
    id="ReachTarget-v0",
    entry_point=lambda **kw: ReachTargetEnv(**kw),
    max_episode_steps=200,
)

# Quick smoke test: run a few random steps.
env = gym.make("ReachTarget-v0")
obs, info = env.reset(seed=42)
print(f"Initial observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

total_reward = 0.0
for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, info = env.reset()
        break

print(f"Total reward after {step + 1} steps: {total_reward:.3f}")
print(f"Final distance to target: {info['distance']:.4f}")
env.close()

# %%
# Step 4 — Key Design Tips
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When creating your own MuJoCo environment, keep these guidelines in mind:
#
# **Observation design**:
#   - Use ``cos`` / ``sin`` of joint angles rather than raw angles
#     to avoid discontinuities at ±π.
#   - Include joint velocities for reactive control.
#   - Express goals as relative vectors (fingertip → target)
#     rather than absolute positions.
#
# **Reward shaping**:
#   - Dense rewards (e.g., negative distance) train faster than
#     sparse rewards (e.g., +1 only on success).
#   - Add a small control penalty to encourage efficient motion.
#   - Consider curriculum learning: start with the target close,
#     then gradually increase the workspace.
#
# **Termination vs. truncation**:
#   - ``terminated=True``: the task is done (reached goal, fell over).
#   - ``truncated=True``: time limit hit (set via ``max_episode_steps``).
#   - Gymnasium handles truncation automatically when you set
#     ``max_episode_steps`` at registration time.
#
# **Performance**:
#   - ``frame_skip`` controls the ratio of simulation steps to
#     control steps. Higher values run faster but limit control frequency.
#   - Cache site/body IDs in ``__init__`` instead of looking them up
#     every step.
#
# **Next steps**:
#   - Train an RL agent (PPO, SAC) on your environment using
#     Stable-Baselines3 or CleanRL.
#   - Use ``render_mode="rgb_array"`` to record evaluation videos.
#   - Load more complex robot models from
#     `MuJoCo Menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_.
