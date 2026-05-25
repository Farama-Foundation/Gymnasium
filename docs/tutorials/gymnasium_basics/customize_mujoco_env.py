"""Customize MuJoCo environment parameters.

This tutorial shows how to customize an existing MuJoCo environment using
keyword arguments passed to ``gym.make``. We will use ``HalfCheetah-v5`` as the
example environment and focus on changing the control cost weight.

The tutorial covers:

1. Creating a default HalfCheetah environment.
2. Inspecting the action space, observation space, and info dictionary.
3. Changing ``ctrl_cost_weight``.
4. Comparing reward components using the same seed and same action.
5. Discussing how the parameter can affect training behavior.

Changing environment parameters can affect learning behavior and benchmark
comparability, so these changes should be made intentionally.
"""

import gymnasium as gym

# %%
# Create the default HalfCheetah environment
# ------------------------------------------
#
# ``HalfCheetah-v5`` is a MuJoCo environment where the agent controls a
# two-dimensional cheetah-like robot. The goal is to move forward while avoiding
# unnecessarily large actions.
#
# We can start by creating the default environment and inspecting its spaces.

env = gym.make("HalfCheetah-v5")

obs, info = env.reset(seed=123)

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
print("Info keys after reset:", info.keys())

env.close()


# %%
# Understanding the action and observation spaces
# -----------------------------------------------
#
# The action space for ``HalfCheetah-v5`` is:
#
# ``Box(-1.0, 1.0, (6,), float32)``
#
# This means the agent chooses six continuous action values, each between
# ``-1.0`` and ``1.0``. These actions correspond to torques applied to the
# HalfCheetah joints.
#
# The observation space is:
#
# ``Box(-inf, inf, (17,), float64)``
#
# This means each observation is a 17-dimensional vector describing the
# simulated robot's current state.


# %%
# Step through the environment
# ----------------------------
#
# After calling ``env.step(action)``, the environment returns:
#
# - ``obs``: the next observation
# - ``reward``: the scalar reward
# - ``terminated``: whether the episode ended because of a terminal condition
# - ``truncated``: whether the episode ended because of a time limit
# - ``info``: additional diagnostic information
#
# For ``HalfCheetah-v5``, the ``info`` dictionary includes useful reward
# components such as ``reward_forward`` and ``reward_ctrl``.

env = gym.make("HalfCheetah-v5")

obs, info = env.reset(seed=123)

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep {step}")
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

env.close()


# %%
# Customize the control cost weight
# ---------------------------------
#
# The total reward in ``HalfCheetah-v5`` is based on two main components:
#
# - ``reward_forward``: rewards the agent for moving forward
# - ``reward_ctrl``: penalizes the agent for using large actions
#
# The ``ctrl_cost_weight`` parameter controls how strongly large actions are
# penalized. A larger value encourages the agent to use smaller, more efficient
# actions.
#
# To make a clean comparison, we will use the same seed and the same action in
# two environments. The only difference will be the value of
# ``ctrl_cost_weight``.


base_env = gym.make("HalfCheetah-v5")
base_env.reset(seed=123)
action = base_env.action_space.sample()
base_env.close()

for cost_weight in [0.05, 0.1, 1.0]:
    env = gym.make("HalfCheetah-v5", ctrl_cost_weight=cost_weight)
    obs, info = env.reset(seed=123)

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nctrl_cost_weight={cost_weight}")
    print("Reward:", reward)
    print("reward_forward:", info["reward_forward"])
    print("reward_ctrl:", info["reward_ctrl"])

    env.close()


# %%
# Compare the reward components
# -----------------------------
#
# In the example above, the same initial seed and same action are used for both
# environments. This means the forward reward should stay the same, while the
# control penalty changes.
#
# Example output:
#
# .. code-block:: text
#
#    ctrl_cost_weight=0.05
#    Reward: 0.16536146457629186
#    reward_forward: 0.25364641155438405
#    reward_ctrl: -0.08828495
#
#    ctrl_cost_weight=0.1
#    Reward: 0.07707651759819967
#    reward_forward: 0.25364641155438405
#    reward_ctrl: -0.1765699
#
#    ctrl_cost_weight=1.0
#    Reward: -1.5120524982051373
#    reward_forward: 0.25364641155438405
#    reward_ctrl: -1.7656989
#
# Notice that ``reward_forward`` is the same in each run, while
# ``reward_ctrl`` scales with ``ctrl_cost_weight``. Increasing
# ``ctrl_cost_weight`` makes the same action more costly, which lowers the
# total reward.
#
# This shows that ``ctrl_cost_weight`` changes the reward structure without
# requiring changes to the environment source code.

# %%
# Training behavior analysis
# --------------------------
#
# Changing ``ctrl_cost_weight`` changes the reward signal used during training,
# so it can affect both the scale of rewards and the type of policy the agent
# learns.
#  As an illustrative experiment, the same SAC configuration was trained on
# ``HalfCheetah-v5`` for 50,000 timesteps using three different values of
# ``ctrl_cost_weight``. The algorithm, training timesteps, evaluation procedure,
# and set of random seeds were kept fixed.
#
# .. list-table::
#    :header-rows: 1
#
#    * - ``ctrl_cost_weight``
#      - seeds
#      - mean evaluation reward
#      - standard deviation across seeds
#    * - ``0.05``
#      - ``3``
#      - ``1848.54``
#      - ``646.98``
#    * - ``0.1``
#      - ``3``
#      - ``2032.82``
#      - ``215.76``
#    * - ``1.0``
#      - ``3``
#      - ``1221.41``
#      - ``175.09``
#
# In this small multi-seed comparison, ``ctrl_cost_weight=0.1`` achieved the
# highest average evaluation reward. The lower value, ``0.05``, produced strong
# results for some seeds but had higher variability across runs. The larger
# value, ``1.0``, consistently produced lower evaluation rewards, suggesting
# that a much stronger control penalty can make learning more conservative in
# this setup.
#
# These results are illustrative rather than a full benchmark, but they show
# that changing ``ctrl_cost_weight`` can affect training behavior in a
# non-trivial way. Since this parameter changes the reward function, customized
# environment results should be reported separately from standard benchmark
# results.


# %%
# Full example
# ------------
#
# The following function combines the main ideas from this tutorial into a
# single runnable example.


def main():
    """Compare HalfCheetah reward components with different control costs."""
    base_env = gym.make("HalfCheetah-v5")
    base_env.reset(seed=123)
    action = base_env.action_space.sample()
    base_env.close()

    for cost_weight in [0.05, 0.1, 1.0]:
        env = gym.make("HalfCheetah-v5", ctrl_cost_weight=cost_weight)
        obs, info = env.reset(seed=123)

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nctrl_cost_weight={cost_weight}")
        print("Reward:", reward)
        print("reward_forward:", info["reward_forward"])
        print("reward_ctrl:", info["reward_ctrl"])

        env.close()


if __name__ == "__main__":
    main()


# %%
# Epilogue
# --------
#
# Customizing MuJoCo environment parameters is useful when experimenting with
# reward design, simulation settings, and learning behavior. In this tutorial,
# we changed ``ctrl_cost_weight`` to control how strongly the environment
# penalizes large actions.
#
# Similar keyword arguments can be passed to other MuJoCo environments depending
# on which parameters they expose. When changing these settings, keep in mind
# that environment parameter changes can affect reproducibility and make results
# harder to compare against standard benchmarks.
#
