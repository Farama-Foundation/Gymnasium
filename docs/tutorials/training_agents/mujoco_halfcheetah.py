# fmt: off
"""
Getting Started with HalfCheetah-v5
=====================================

.. image:: /_static/img/tutorials/halfcheetah.gif
  :width: 400
  :alt: halfcheetah-running

HalfCheetah is one of the most commonly used MuJoCo benchmark environments.
The goal is to make a 2D cheetah-like robot run as fast as possible by applying
torques to its joints.

This tutorial walks through:

- Loading and inspecting the HalfCheetah-v5 environment
- Understanding the observation and action spaces
- Running a random policy to see what the environment looks like
- A simple training loop structure you can build on

**Environment basics:**

- **Observation space**: 17-dimensional vector (joint positions, velocities)
- **Action space**: 6-dimensional continuous vector (torques on joints), clipped to [-1, 1]
- **Reward**: forward velocity minus control cost
- **Episode end**: truncated after 1000 steps (no termination condition)

More details: https://gymnasium.farama.org/environments/mujoco/half_cheetah/

"""
from __future__ import annotations

import numpy as np

import gymnasium as gym


# %%
# Creating the environment
# -------------------------
# ``render_mode="rgb_array"`` lets us capture frames without opening a window.

env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
obs, info = env.reset(seed=42)

print("Observation space:", env.observation_space)
print("Action space:     ", env.action_space)
print("Obs shape:        ", obs.shape)


# %%
# Observation breakdown
# ----------------------
# The 17 values break down as:
#
# - indices 0:8   — joint angles (qpos, excluding root x which is excluded by default)
# - indices 8:17  — joint velocities (qvel)
#
# You can inspect the raw MuJoCo data via ``env.unwrapped.data``.

print("\nFirst obs:", np.round(obs, 3))


# %%
# Random policy rollout
# ----------------------
# Before training anything, it helps to run a random policy to get a feel
# for the reward scale and episode length.

total_reward = 0
steps = 0

obs, info = env.reset(seed=0)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        break

print(f"\nRandom policy — steps: {steps}, total reward: {total_reward:.2f}")
# Expect a low or negative reward since random torques waste energy


# %%
# Reward structure
# -----------------
# The reward at each step is:
#
# .. code-block:: text
#
#   reward = forward_reward - ctrl_cost
#
# where:
#
# - ``forward_reward = velocity_x * dt`` — reward for moving right
# - ``ctrl_cost = 0.1 * sum(action**2)`` — penalty for large torques
#
# You can inspect reward components via the ``info`` dict:

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("\nReward components:", info)


# %%
# Training loop skeleton
# -----------------------
# HalfCheetah has a continuous action space, so you need an actor that outputs
# a continuous distribution (e.g. Gaussian). Policy gradient methods like
# PPO or SAC work well here. Below is the bare skeleton of a training loop:
#
# .. code-block:: python
#
#   for episode in range(num_episodes):
#       obs, _ = env.reset()
#       done = False
#       while not done:
#           action = policy(obs)          # your policy here
#           obs, reward, terminated, truncated, info = env.step(action)
#           store_transition(obs, action, reward)
#           done = terminated or truncated
#       update_policy()
#
# For a full working example using REINFORCE on InvertedPendulum (a simpler
# MuJoCo env), see ``mujoco_reinforce.py`` in this same directory.


env.close()
