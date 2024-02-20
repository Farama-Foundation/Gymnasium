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
__date__ = "2023-02-20"
__license__ = "MIT License"

import copy
import random
from collections import namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import gymnasium


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


# %%
# Helper functions/classes
# ------------------------
# All the functions we need to perform small but repetitive tasks
# that are not related to the agent/training logic.
class Results:
    """
    This class stores results from episodes of agent-environment interaction and
    uses these results for reporting progress or overall agent performance.
    """

    __slots__ = []

    def __init__(self):
        pass


def to_tensor(array: np.ndarray, normalize: bool = False) -> torch.Tensor:
    """
    Takes any array-like object and turns it into a torch.Tensor on `device`.
    For atari image observations, the normalize parameter can be used to change
    the range from [0,255) to [0,1).

    Args:
        array: An array, which can be states, actions, rewards, etc.
        normalize: Whether to normalize image observations

    Returns:
        tensor
    """
    tensor = torch.tensor(array, device=device)

    if normalize:
        return tensor / 255.0
    return tensor


def set_seed(seed):
    # TODO: implement this function to seed all random libraries such as random, numpy, torch.
    pass


def create_atari_env(env: gymnasium.Env) -> gymnasium.Env:
    # TODO: implement function and write doc
    return env


# %%
# Agent
# -----
# What is the agent, how is it defined, what can it do, why do we do it as we do?
# Write a short description.


class ReplayMemory:
    """
    Buffer for experience storage.

    This implementation uses numpy arrays for storing experiences, and allocating
    required RAM upfront instead of storing dynamically. This enables us to know
    upfront whether enough memory is available on the machine, insteaf of the training
    being quit unexpectedly.
    Each term in an experience (state, action , next_state, reward ,done) is stored into
    separate arrays. A maximum capacity is required upon initialization, as well as
    observation shape, and whether the observation is image data, such as in Atari games.

    In Atari games, the frame stacking technique is used, where the past four observations make
    up a state. Thus, for each experience, `state` and `next_state` are four frames each, however
    the first three frames of `next_state` is the last three frames of `state`, hence these frames
    are stored once in the `next_states` array, and when sampling, reconcatenated back to build a
    proper `next_state`.
    """

    def __init__(self, params: namedtuple) -> None:
        """
        Initialize a replay memory.

        Args:
            params: A namedtuple containing all hperparameters needed for training an agent,
                    hence it contains all the parameters needed for creating a memory buffer.
        """
        self.params = params
        self.length: int = 0  # number of experiences stored so far
        self.index: int = 0  # current index to store data to

        # shape of state buffers differ depending on whether an obs is image data.
        if self.params.image_obs:
            self._state_shape = (
                self.params.capacity,
                self.params.frame_stacking,
                *self.params.obs_shape,
            )
            self._next_state_shape = (
                self.params.capacity,
                1,
                *self.params.obs_shape,
            )
        else:
            self._state_shape = (self.params.capacity, *self.params.obs_shape)
            self._next_state_shape = (self.params.capacity, *self.params.obs_shape)

        # creating the buffers
        self._states: np.ndarray = np.zeros(shape=self._state_shape, dtype=np.float16)
        self._actions: np.ndarray = np.zeros(self.params.capacity, dtype=np.uint8)
        self._next_states: np.ndarray = np.zeros(
            shape=self._next_state_shape, dtype=np.float16
        )
        self._rewards: np.ndarray = np.zeros(self.params.capacity, dtype=np.float16)
        self._dones: np.ndarray = np.zeros(self.params.capacity, dtype=np.uint8)

    def __len__(self):
        """Returns the length of the memory."""
        return self.length

    def push(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """
        Adds a new experience into the buffer

        Args:
            state: Current agent state.
            action: The taken action at `state`.
            next_state: The resulting state from taking action.
            reward: Reward signal.
            done: Whether episode endeed after taking action.

        Returns:

        """
        self._states[self.index] = state
        self._actions[self.index] = action
        self._next_states[self.index] = next_state
        self._rewards[self.index] = reward
        self._dones[self.index] = done

        self.length = min(self.length + 1, self.params.capacity)
        self.index = (self.index + 1) % self.params.capacity

        return

    def sample(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples and returns a random batch of experiences.
        """
        indices = np.random.choice(
            a=np.arange(self.length), size=self.params.batch_size, replace=False
        )

        states = self._states[indices]
        next_states = self._next_states[indices]
        next_states = (
            self._next_states[indices]
            if self.params.image_obs
            else np.concatenate((states[:, 1:, :, :], next_states), axis=1)
        )

        actions = self._actions[indices]
        rewards = self._rewards[indices]
        dones = self._dones[indices]

        if self.params.image_obs:
            assert torch.equal(
                states[:, 1:, :], next_states[:, :3, :]
            ), "Incorrect concatenation."

        return states, actions, next_states, rewards, dones


class Agent(nn.Module):
    """
    Class for agent running on Categorical-DQN (C51) algorithm.
    In essence, for each action, a value distribution is returned,
    from which a statistic such as the mean is computedto get the
    action-value.
    """

    def __init__(self, params: namedtuple):
        """
        Initializing the agent class.

        Args:
            params: A namedtuple containing the hyperparameters.
        """
        super().__init__(params)
        self.params = params
        self.epsilon = self.params.eps_start
        self.eps_reduction = (self.params.eps_start - self.params.eps_end) / (
            self.params.anneal_length_percent * self.params.training_steps
        )

        self.reduce_eps_steps = (
            self.params.eps_start - self.params.eps_end
        ) / self.params.anneal_length
        self.delta = (self.params.v_max - self.params.v_min) / (self.params.n_atoms - 1)

        self.replay_memory = ReplayMemory(self.params)

        self.optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.params.learning_rate
        )

        # The support is the set of values over which a probability
        # distribution is defined and has non-zero probability there.
        self.support = torch.linspace(
            start=self.params.v_min, end=self.params.v_max, steps=self.params.n_atoms
        ).to(device)

        # -- defining the neural network --
        in_features = self.params.in_features
        out_features = self.params.n_actions * self.params.n_atoms
        n_hidden_units = self.params.n_hidden_units

        # the convolutional part is created depending on whether the input is image observation.
        # These convolutional layers is in accordance to the DQN network parameters.
        if self.params.image_obs:
            self.conv1 = nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=(8, 8),
                stride=(4, 4),
                padding=0,
            )
            self.conv2 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(0, 0),
            )
            self.conv3 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(0, 0),
            )
        else:
            self.conv1, self.conv2, self.conv3 = None, None, None

        self.fc1 = nn.Linear(in_features=in_features, out_features=n_hidden_units)
        self.fc2 = nn.Linear(in_features=n_hidden_units, out_features=out_features)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the agent network.
        Args:
            state: Current state of the environment. Single or multiple states stacked.

        Returns:
            value_dist: Tensor of action value-distribution for each action and state.
                    Values are softmax probabilities for each action.
                    shape=(n_states, n_actions, n_atoms)

        """
        if self.params.image_obs:
            conv1_out = nn.ReLU(self.conv1(state))
            conv2_out = nn.ReLU(self.conv2(conv1_out))
            conv3_out = nn.Flatten(nn.ReLU(self.conv3(conv2_out)))
            fc1_out = nn.ReLU(self.fc1(conv3_out))
            value_dist = self.fc2(fc1_out)
        else:
            fc1_out = nn.ReLU(self.fc1(state))
            value_dist = self.fc2(fc1_out)

        value_dist = value_dist.view(state.shape[0], self.n_actions, self.N).softmax(
            dim=2
        )
        return value_dist

    def act(self, state: torch.Tensor, exploit: bool) -> int:
        """
        Sampling action for a given state. Actions are sampled randomly during exploration.
        The action-value is the expected value of the action value-distribution.

        Args:
            state: Current state of agent.
            exploit: True when not exploring.

        Returns:
            action: The sampled action.
        """
        random_value = random.random()

        if self.epsilon > self.params.eps_end:
            self.epsilon -= self.eps_reduction

        with torch.no_grad():
            value_dist = self.forward(state)

        expected_returns = torch.sum(self.support * value_dist, dim=2)

        if exploit or random_value > self.epsilon:
            action = torch.argmax(expected_returns, dim=1).item()
        else:
            action = torch.randint(
                high=self.params.n_actions, size=(1,), device=device
            ).item()

        return action

    def store_experience(
        self, state, action: int, next_state, reward: float, done: bool
    ):
        """

        Args:
            state: Latest agent state.
            action: Action taken at latest state
            next_state: Resulting state after taking action.
            reward: Received reward signal.
            done: Whether the action terminated the episode.

        Returns:

        """
        # TODO: fix the type hints for the function parameters state and next_state
        self.replay_memory.push(
            state=state, action=action, next_state=next_state, reward=reward, done=done
        )

    def learn(self, target_agent: "Agent") -> float:
        """
        Learning steps, which includes updating the network parameters through backpropagation.
        Args:
            target_agent: The target agent used for storing previous learning step network parameters.

        Returns:
            loss: The loss, which is defined as the expected difference between
                  the agent itself and the `target_agent` predictions on a batch
                  of states.
        """
        states, actions, next_states, rewards, dones = self.replay_memory.sample()

        states = to_tensor(array=states, normalize=self.params.image_obs)
        actions = to_tensor(array=actions).view(-1, 1)
        next_states = to_tensor(array=next_states, normalize=self.params.image_obs)
        rewards = to_tensor(array=rewards).view(-1, 1)
        dones = to_tensor(array=dones).view(-1, 1)

        # agent predictions
        value_dists = self.forward(states)
        # gather probs for selected actions
        probs = value_dists[torch.arange(self.params.batch_size), actions.view(-1), :]

        # target agent predictions
        with torch.no_grad():
            target_value_dists = target_agent.forward(next_states)

        # ------------------------------ Categorical algorithm ------------------------------
        #
        # Since we are dealing with value distributions and not value functions,
        # we can't minimize the loss using MSE(reward+gamma*Q_i-1 - Q_i). Instead,
        # we project the support of the target predictions T_hat*Z_i-1 onto the support
        # of the agent predictions Z_i, and minimize the cross-entropy term of
        # KL-divergence `KL(projected_T_hat*Z_i-1 || Z_i)`.
        #

        next_actions = ((self.forward(next_states) * self.support).sum(dim=2)).argmax(
            dim=1
        )
        target_probs = target_value_dists[
            torch.arange(self.params.batch_size), next_actions, :
        ]

        m = torch.zeros(self.params.batch_size * self.params.n_atoms).to(device)

        Tz = (rewards + self.params.gamma * self.support).clip(
            self.params.v_min, self.params.v_max
        )
        bj = (Tz - self.params.v_min) / self.delta

        l, u = torch.floor(bj).long(), torch.ceil(bj).long()

        offset = (
            torch.linspace(
                start=0,
                end=(self.params.batch_size - 1) * self.params.n_atoms,
                steps=self.params.batch_size,
            )
            .long()
            .unsqueeze(1)
            .expand(self.params.batch_size, self.N)
            .to(device)
        )
        m.index_add_(0, (l + offset).view(-1), (target_probs * (u - bj)).view(-1))
        m.index_add_(0, (u + offset).view(-1), (target_probs * (bj - l)).view(-1))

        m = m.view(self.params.batch_size, self.params.n_atoms)
        # -----------------------------------------------------------------------------------

        loss = (-(m * torch.log(probs).sum(dim=1))).mean()

        # set all gradients to zero
        self.optimizer.zero_grad()

        # backpropagate loss through the network
        loss.backward()

        # update weights
        self.optimizer.step()

        return loss.item()


# %%
# Training
# --------
# Describe how training is performed, what is needed, mention that hyperparameters will be explained later.
# Describe useful statistics to plot during training to track agent progress.


def train(params: namedtuple, seed: int, verbose: bool = True):
    """
    Creates agent and environment, and lets the agent interact
    with the environment until it learns a good policy.

    Args:
        params: A namedtuple containing all necessary hyperparameters.
        seed: For reprodicubility.
        verbose: Whether to print training progress periodically.

    Returns:

    """
    set_seed(seed)

    steps = 0  # global time steps for the whole training

    # results_collector = Results()

    env = gymnasium.make(params.env_name)
    if params.image_obs:
        env = create_atari_env(env)

    agent = Agent(params=params)
    target_agent = copy.deepcopy(agent)
    # Q_target parameters are frozen.
    for p in target_agent.parameters():
        p.requires_grad = False

    while steps < params.training_steps:
        done = False
        obs, info = env.reset(seed=seed)

        while not done:
            state = to_tensor(obs, params.image_obs)
            action = agent.act(state=state, exploit=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            agent.store_experience(
                state=obs, action=action, next_state=next_obs, reward=reward, done=done
            )
            obs = next_obs

            # TODO: store results

            steps += 1

            # train agent periodically if enough experience exists
            if steps % params.update_frequency == 0 and (
                len(agent.replay_memory) > params.batch_size
            ):
                _ = agent.learn(target_agent)

            # Update the target network periodically.
            if steps % params.target_update_frequency == 0:
                target_agent.load_state_dict(agent.state_dict())

            # print progress periodically
            if verbose and steps % 10_000 == 0:
                print(f"step={steps}/{params.training_steps}")

    # TODO: clean up, create results summary, return stuff
    env.close()

    return


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
Hyperparameters = namedtuple(
    "Hyperparameters",
    [
        # --- env related ---
        "env_name",
        "n_actions",
        # --- training related ---
        "training_steps",  # number of steps to train agent for
        "obs_shape",  # a tuple representing shape of observations, ex. (1, 4), (4, 84)
        "image_obs",  # boolean, indicating whether the env provides image observations
        "batch_size",  # number of experiences to sample for updating agent network parameters
        "update_frequency",  # how often to update agent network parameters
        "target_update_frequency",  # how often to replace target agent network parameters
        "gamma",  # discount factor
        "frame_stacking",  # number of frames to be stacked together
        # --- exploration-exploitation strategy related ---
        "epsilon_start",
        "epsilon_end",
        "anneal_length_percentage",
        # --- neural network related ---
        "in_features",
        "n_hidden_units",
        # --- optimizer related ---
        "learning_rate",
        # --- replay memory related ---
        "capacity",
        # --- agent algorithm related ---
        "v_min",
        "v_max",
        "n_atoms",
    ],
)

# %%
# Env1
# ---------------
# Define hyperparameters dict.
# train for different seeds.
# evaluate all agents.
# plot the progress and evaluation.
# env1_hyperparameters = Hyperparameters()


env = None


# %%
# Env2
# ---------------
# Define hyperparameters dict.
# train for different seeds.
# evaluate all agents.
# plot the progress and evaluation.
# env2_hyperparameters = Hyperparameters()


env2 = None

# AtariPreprocessing
# FrameStack
# RecordEpisodeStatistics
# 512 hidden units for atari
# frame-skipping done by default in AtariPreprocessing

# RecordVideo
# how to evaluate a trained agent

# %%
# Finishing words
# ---------------
# whatever is remaining to be said.


# =========================================== END OF FILE ===========================================
