"""
Implementing Custom Wrappers
============================

In this tutorial we will describe how to implement your own custom wrappers.

Wrappers are a great way to add functionality to your environments in a modular way.
This will save you a lot of boilerplate code.

We will show how to create a wrapper by

- Inheriting from :class:`gymnasium.ObservationWrapper`
- Inheriting from :class:`gymnasium.ActionWrapper`
- Inheriting from :class:`gymnasium.RewardWrapper`
- Inheriting from :class:`gymnasium.Wrapper`

Before following this tutorial, make sure to check out the docs of the :mod:`gymnasium.wrappers` module.
"""

# %%
# Inheriting from :class:`gymnasium.ObservationWrapper`
# -----------------------------------------------------
# Observation wrappers are useful if you want to apply some function to the observations that are returned
# by an environment. If you implement an observation wrapper, you only need to define this transformation
# by implementing the :meth:`gymnasium.ObservationWrapper.observation` method. Moreover, you should remember to
# update the observation space, if the transformation changes the shape of observations (e.g. by transforming
# dictionaries into numpy arrays, as in the following example).
#
# Imagine you have a 2D navigation task where the environment returns dictionaries as observations with
# keys ``"agent"`` and ``"target"`` (like the ``GridWorldEnv`` from :doc:`environment_creation`). A common thing to do
# might be to throw away some degrees of freedom and only consider the position of the target relative to the agent,
# i.e. ``observation["target"] - observation["agent"]``. For this, you could implement an observation wrapper like
# this:

import numpy as np

import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
from gymnasium.spaces import Box, Discrete


class RelativePosition(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            shape=(2,), low=-np.inf, high=np.inf, dtype=np.float32
        )

    def observation(self, obs):
        # cast to the dtype of the new observation space (the positions are integers)
        return (obs["target"] - obs["agent"]).astype(np.float32)


# %%
# Inheriting from :class:`gymnasium.ActionWrapper`
# ------------------------------------------------
# Action wrappers can be used to apply a transformation to actions before applying them to the environment.
# If you implement an action wrapper, you need to define that transformation by implementing
# :meth:`gymnasium.ActionWrapper.action`. Moreover, you should specify the domain of that transformation
# by updating the action space of the wrapper.
#
# Let’s say you have an environment with action space of type :class:`gymnasium.spaces.Box`, but you would only like
# to use a finite subset of actions. Then, you might want to implement the following wrapper:


class DiscreteActions(ActionWrapper):
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, act):
        return self.disc_to_cont[act]


env = gym.make("LunarLanderContinuous-v3")
print(env.action_space)  # Box(-1.0, 1.0, (2,), float32)
wrapped_env = DiscreteActions(
    env, [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
)
print(wrapped_env.action_space)  # Discrete(4)


# %%
# Inheriting from :class:`gymnasium.RewardWrapper`
# ------------------------------------------------
# Reward wrappers are used to transform the reward that is returned by an environment.
# As for the previous wrappers, you need to specify that transformation by implementing the
# :meth:`gymnasium.RewardWrapper.reward` method.
#
# Let us look at an example: Sometimes (especially when we do not have control over the reward
# because it is intrinsic), we want to clip the reward to a range to gain some numerical stability.
# To do that, we could, for instance, implement the following wrapper:

from typing import SupportsFloat


class ClipReward(RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, r: SupportsFloat) -> SupportsFloat:
        return np.clip(r, self.min_reward, self.max_reward)


# %%
# Inheriting from :class:`gymnasium.Wrapper`
# ------------------------------------------
# Sometimes you might need to implement a wrapper that does some more complicated modifications (e.g. modify the
# reward based on data in ``info`` or change the rendering behavior).
# Such wrappers can be implemented by inheriting from :class:`gymnasium.Wrapper`.
#
# - You can set a new action or observation space by defining ``self.action_space`` or ``self.observation_space`` in ``__init__``, respectively
# - You can set new metadata by defining ``self.metadata`` in ``__init__``
# - You can override :meth:`gymnasium.Wrapper.step`, :meth:`gymnasium.Wrapper.render`, :meth:`gymnasium.Wrapper.close` etc.
#
# If you do this, you can access the environment that was passed
# to your wrapper (which *still* might be wrapped in some other wrapper) by accessing the attribute :attr:`env`.
#
# Let's also take a look at an example for this case. *Reacher* returns a dense reward: the negative distance between
# its fingertip and the target, plus a penalty for large actions (the weights of both terms can be set with the
# ``reward_dist_weight`` and ``reward_control_weight`` arguments of ``Reacher-v5``). Suppose that we instead want a sparse
# goal-reaching task: a reward of 1 when the fingertip is within some distance of the target, with the episode
# terminating at that point. This changes the reward, the termination signal and the ``info`` at once, so we override
# :meth:`gymnasium.Wrapper.step`. The last two entries of Reacher's observation are the fingertip position relative to
# the target:


class ReacherSparseReward(Wrapper):
    def __init__(self, env, distance_threshold=0.02):
        super().__init__(env)
        self.distance_threshold = distance_threshold

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        distance = np.linalg.norm(obs[-2:])
        success = bool(distance < self.distance_threshold)

        info["is_success"] = success
        reward = 1.0 if success else 0.0
        return obs, reward, terminated or success, truncated, info


env = ReacherSparseReward(gym.make("Reacher-v5"))
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(reward, terminated, info["is_success"])

# %%
# Wrappers without a subclass, and vector wrappers
# ------------------------------------------------
# For a simple transformation, you don't need to write a class: :class:`gymnasium.wrappers.TransformObservation`,
# :class:`gymnasium.wrappers.TransformAction` and :class:`gymnasium.wrappers.TransformReward` take a function
# (for example, a ``lambda``) and apply it to the observations, actions or rewards:

from gymnasium.wrappers import TransformReward

env = TransformReward(gym.make("CartPole-v1"), lambda r: 0.01 * r)

# %%
# Vector environments have their own wrapper base classes, :class:`gymnasium.vector.VectorWrapper`,
# :class:`gymnasium.vector.VectorObservationWrapper`, :class:`gymnasium.vector.VectorActionWrapper` and
# :class:`gymnasium.vector.VectorRewardWrapper`, which work the same way on batches of observations, actions and
# rewards. See :doc:`/api/vector/wrappers` for the vector versions of the built-in wrappers.
