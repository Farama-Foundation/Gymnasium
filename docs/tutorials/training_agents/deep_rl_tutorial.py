"""
Implementation of a deep RL agent
=================================

intro: what is this tutorial and why. what will it contain

In this tutorial we describe and show how a deep reinforcement learning agent is implemented and trained using pytorch
v2.0.1. The agent is C51. But there is not much difference between different agents, they all are implemented similarly.

Let's start by importing necessary libraries:
"""

# Global TODOs:
# TODO: continue with replay memory
# TODO:
# TODO: Final check on documentation and typing.

# Good to know:
#   - Use atari preprocessing wrapper
#   - Use env.spec.max_episode_steps
#   - Use framestack wrapper with compression. When storing to memory, use only last one
#   - Use record episode statistics wrapper to store episode return and length.
#   - Use gymnasium.utils.save_video.save_video
#   - Check gradient clipping # In-place clipping: torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
#   - for the agent, done = terminated, even though an episode can end due to timelimit or other things
#   -
#
#   - Keep track of useful statistics for debugging, such as return, length, td-error, ..


# %%
__author__ = "Hardy Hasan"
__date__ = "2023-10-08"
__license__ = "MIT License"

# import copy
# import random
# import time
from typing import Tuple

import numpy
import numpy as np
import torch


# import torch.nn as nn

# import gymnasium as gym


# utilize gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Skeleton
# --------
#
# what classes and functions are required to implement an agent.
# train, evaluate, agent, environment
#
# The essential parts of an implementation is an environment that we want to solve, an agent that will be trained
# to solve the environment, a training function that will be used to train the agent and an evaluation function
# to evaluate the agent when its training has finished.
#
# The agent itself is an object that has the ability to act given observation, a memory to store experiences to and,
# sample experiences from, a mechanism to learn from experience.
#
# Let's start by implementing the memory and agent:


class ReplayMemory:
    """
    Buffer for experience storage.

    This implementation uses numpy arrays for storing experiences, thus allocating
    required RAM upfront instead of storing dynamically. This will enable us to know
    upfront whether enough memory is available on the machine, insteaf of the training
    being quit unexpectedly.
    Each term in an experience (state, action , next_state, reward ,done) is stored separately
    into a different array. A maximum capacity is required upon initialization, as well as
    observation shape, and whether the observation is image data, such as in Atari games.

    In Atari games, the frame stacking technique is used, where the past four observations make
    up a state. Thus, for each experience, `state` and `next_state` are four frames each, however
    the first three frames of `next_state` is the last three frames of `state`, hence these frames
    are stored once in the states array, and when sampling, reconcatenated back to build a proper
    `next_state`.
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        obs_shape: Tuple[int, int],
        image_obs: bool,
        frame_stacking: int = 4,
    ) -> None:
        """
        Initialize a replay memory.
        """
        self._capacity = capacity
        self._batch_size = batch_size
        self._length: int = 0  # number of experiences stored so far
        self._index: int = 0  # current index to store data to

        # shape of state buffers differ depending on whether an obs is image data.
        if image_obs:
            self._state_shape = (self._capacity, frame_stacking, *obs_shape)
            self._next_state_shape = (
                self._capacity,
                1,
                *obs_shape,
            )  # storing one frame for next_state
            self._dtype = (
                np.uint8
            )  # image data are integers, hence using this data type.
        else:
            self._state_shape = (self._capacity, *obs_shape)
            self._next_state_shape = (self._capacity, *obs_shape)
            self._dtype = (
                np.float16
            )  # other obs can be non-integers, hence using this data type.

        # creating the buffers
        self._states: np.ndarray = np.zeros(shape=self._state_shape, dtype=self._dtype)
        self._actions: np.ndarray = np.zeros(self._capacity, dtype=np.uint8)
        self._next_states: np.ndarray = np.zeros(
            shape=self._next_state_shape, dtype=self._dtype
        )
        self._rewards: np.ndarray = np.zeros(self._capacity, dtype=np.float16)
        self._dones: np.ndarray = np.zeros(self._capacity, dtype=np.uint8)

    def __len__(self):
        """Returns the length of the memory."""
        return self._length

    def push(
        self,
        state: numpy.ndarray,
        action: int,
        next_state: numpy.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Adds a new experience into the buffer"""
        self._states[self._index] = state
        self._actions[self._index] = action
        self._next_states[self._index] = next_state
        self._rewards[self._index] = reward
        self._dones[self._index] = done

        self._length = min(self._length + 1, self._capacity)
        self._index = (self._index + 1) % self._capacity

        return

    def sample(self):
        """
        Samples a random batch of experiences and returns them as torch.Tensors.
         -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # indices = np.random.choice(a=np.arange(self._length), size=self._batch_size, replace=False)

        # states = self._states[indices]
        # actions = self._actions[indices]
        # next_states = np.concatenate(states[:, 1:, :, :], self._next_states[indices]), axis=1)
        # rewards = self._rewards[indices]
        # dones = self._dones[indices]


class Agent:
    pass


# %%
# Training
# --------
# Describe how training is performed, what is needed, mention that hyperparameters will be explained later.
# Describe useful statistics to plot during training to track agent progress.


def train():
    pass


# %%
# Evaluation
# ----------
# Describe how an agent should be evaluated once its training is finished.


def evaluate():
    pass


# %%
# Hyperparameters
# ---------------
# What hyperparameters are necessary, what are common good values etc.
# Describe the importance of seeds.


# %%
# Env1
# ---------------
# Define hyperparameters dict.
# train for different seeds.
# evaluate all agents.
# plot the progress and evaluation.


env = None


# %%
# Env2
# ---------------
# Define hyperparameters dict.
# train for different seeds.
# evaluate all agents.
# plot the progress and evaluation.


env2 = None


# %%
# Finishing words
# ---------------
# whatever is remaining to be said.


# =========================================== END OF FILE ===========================================
