"""
Implementation of a deep RL agent
=================================

intro: what is this tutorial and why. what will it contain

In this tutorial we describe and show how a deep reinforcement learning agent is implemented and trained using pytorch
v2.0.1. The agent is C51. But there is not much difference between different agents, they all are implemented similarly.

Let's start by importing necessary libraries:
"""

# Global TODOs:
# TODO: Finish agent class
# TODO:
# TODO: Final check on documentation and typing.


# %%
__author__ = "Hardy Hasan"
__date__ = "2023-12-11"
__license__ = "MIT License"

# import copy
# import random
# import time
from typing import Tuple

import numpy
import numpy as np
import torch
import torch.nn as nn


# import torch.nn as nn

# import gymnasium as gym


# utilize gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    # TODO: implement this function to seed all random libraries such as random, numpy, torch.
    pass


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

# %%
# Agent
# -----
# What is the agent, how is it defined, what can it do, why do we do it as we do?
# Write a short description.


class ReplayMemory:
    """
    Buffer for experience storage.

    This implementation uses numpy arrays for storing experiences, and allocating
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
        self._image_obs = image_obs
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
        else:
            self._state_shape = (self._capacity, *obs_shape)
            self._next_state_shape = (self._capacity, *obs_shape)

        # creating the buffers
        self._states: np.ndarray = np.zeros(shape=self._state_shape, dtype=np.float16)
        self._actions: np.ndarray = np.zeros(self._capacity, dtype=np.uint8)
        self._next_states: np.ndarray = np.zeros(
            shape=self._next_state_shape, dtype=np.float16
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

    def sample(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random batch of experiences and returns them as torch.Tensors.
        """
        indices = np.random.choice(
            a=np.arange(self._length), size=self._batch_size, replace=False
        )

        states = self._states[indices]
        next_states = self._next_states[indices]
        next_states = (
            np.concatenate((states[:, 1:, :, :], next_states), axis=1)
            if self._image_obs
            else self._next_states[indices]
        )

        states = torch.tensor(states, dtype=torch.float, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=device)
        actions = torch.tensor(
            self._actions[indices], dtype=torch.int64, device=device
        ).view(-1, 1)
        rewards = torch.tensor(
            self._rewards[indices], dtype=torch.float, device=device
        ).view(-1, 1)
        dones = torch.tensor(self._dones[indices], dtype=torch.int, device=device).view(
            -1, 1
        )

        if self._image_obs:
            assert torch.equal(
                states[:, 1:, :], next_states[:, :3, :]
            ), "Incorrect concatenation."

        return states, actions, next_states, rewards, dones


# %%
# Test ReplayMemory
# This section can be removed later when everything is done. It is here for
# debugging purposes only.

# test 1: array obs
test1_seed = 111
np.random.seed(test1_seed)
capacity = 5
batch_size = 3
obs_shape = (1, 3)
image_obs = False
mem = ReplayMemory(
    capacity=capacity, batch_size=batch_size, obs_shape=obs_shape, image_obs=image_obs
)
states = [np.random.random((1, 3)) for _ in range(capacity + 1)]
actions = [np.random.randint(10) for _ in range(capacity + 1)]
next_states = [np.random.random((1, 3)) for _ in range(capacity + 1)]
rewards = [np.random.random() for _ in range(capacity + 1)]
dones = [[True, False][i % 2] for i in range(capacity + 1)]
# make sure that each obs goes into right position, and length and index are correctly updated.
for i in range(capacity + 1):
    mem.push(states[i], actions[i], next_states[i], rewards[i], dones[i])
# make sure that sapling a batch returns correct dimensions
batch = mem.sample()

# test 2: image obs
test1_seed = 111
np.random.seed(test1_seed)
capacity = 5
batch_size = 3
obs_shape = (3, 3)
image_obs = True
mem = ReplayMemory(
    capacity=capacity, batch_size=batch_size, obs_shape=obs_shape, image_obs=image_obs
)
states = [np.random.random((4, 3, 3)) for _ in range(capacity + 1)]
next_states = [np.random.random((1, 3, 3)) for _ in range(capacity + 1)]

for i in range(capacity + 1):
    mem.push(states[i], actions[i], next_states[i], rewards[i], dones[i])
# make sure that sapling a batch returns correct dimensions
batch = mem.sample()


class Agent(nn.Module):
    """Class for Categorical-DQN (C51). For each action, a value distribution is provided."""

    def __init__(self, params):
        """
        Initializing the agent class.
        Args:
            params:
        """
        super().__init__(params)

        self.image_obs = params["image_obs"]
        self.n_atoms = params["n_atoms"]
        self.v_min = params["v_min"]
        self.v_max = params["v_max"]
        self.gamma = params["gamma"]
        self.n_actions = params["n_actions"]
        self.epsilon_end = params["epsilon_end"]

        self.delta = (self.v_max - self.v_min) / (self.n_atoms - 1)

    def forward(self, state):
        """
        Forward pass through the agent network.
        Args:
            state:

        Returns:

        """
        pass

    def act(self, state, exploit):
        """
        Taking action in a given state.
        Args:
            state:
            exploit:

        Returns:

        """
        pass

    def learn(self, state):
        """
        Learning steps, which includes updating the network parameters through backpropagation.
        Args:
            state:

        Returns:

        """
        pass

    def categorical_algorithm(self):
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
