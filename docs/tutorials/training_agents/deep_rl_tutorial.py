"""
Implementation of a Deep RL Agent
=================================

This tutorial serves as a step-by-step guide for implementing a deep reinforcement learning agent from scratch, as well
as training agents on various environments ranging from easy to learn environments such as Cart Pole to more difficult
ones such as the atari environments. It assumes familiarity with RL as well as the DQN agent. Pytorch is used for the
deep learning part. Moreover, the same implementation is used to train agents on all of these environments without the
need to do multiple implementations due to the difference in observations and all other code differences that comes
with it. All is needed is to provide the name of an environment and choosing hyperparameters, then training can be
started.

We will be implementing the Categorical-DQN agent (C51-agent) (Bellemare et al. (2017)), which is a distributional
version of the DQN agent where instead of action-values, the network outputs the entire return distribution for each
action, and actions are chosen based on expected returns. It is worth noting that although one specific agent is
implemented here, the other variants of the DQN agent follow this way of implementation more or less, therefore
implementing another agent based on this structure is straightforward.

The essential components of an implementation are as follows: an *Environment* that we want to solve, an *Agent* which
interacts with the environment during a number of steps to learn its dynamics, a *Function* that enables the agent to
interact with the environment, and additionally functions to visualize the training progress and what the agent has
learned.

This tutorial is structured into a number of sections, where the first contains section a description of the helper
functions, the second section contains the environments creation, the third section contains the agent class,
network class and the replay memory class,  the fourth section contains the functions for training, the fifth section
contains code related to visualization and the sixth and last section displays the results obtained for the
different environments considered.
"""

# %%
__author__ = "Hardy Hasan"
__date__ = "2023-07-01"
__license__ = "MIT License"

import collections
import concurrent.futures
import dataclasses
import random
import time
from collections import namedtuple
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import moviepy.editor
import numpy as np
import torch
import torch.nn as nn

import gymnasium


# utilize gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# 1. Helper classes and functions:
# --------------------------------
# The first thing we need here is a class for defining the whole set of hyperparameters. We also need a class for
# storing the training progress periodically, so that it can be used for tracking performance, debugging and
# visualization. Next, we need a function that takes an iterable or array as input and returns a tensor out of it,
# which can be used to transform observations into torch tensors. We also have a namedtuple for agent's action info.
# Lastly, we need a function to seed all sources of randomness, so that we can reproduce results for debugging,
# but also show that the training is stable across a number of different seeds. Note however that when training
# on `torch.cuda`, there will be some randomness as the things are seeded now, because the choice of algorithm for
# a certain task can be different across runs, even on the same seed and hardware. Although it can be fixed, it
# comes at cost on performance, therefore it is left to be random.
#
@dataclasses.dataclass
class Hyperparameters:
    # --- env related ---
    env_name: str = ""
    n_actions: int = 0

    # --- training related ---
    training_steps: int = 1000  # number of steps to train agent for
    learning_starts: int = 100  # number of steps played before agent learns.
    obs_shape: tuple = (1,)  # of observations, ex. (1, 4), (84, 84)
    image_obs: bool = False  # whether env provides image observations
    batch_size: int = 32  # number of experiences to sample for learning step
    update_frequency: int = 4  # how often to update agent network parameters
    target_update_frequency: int = 1000  # target network update frequency
    gamma: float = 0.99  # discount factor
    num_frame_stacking: int = 1  # number of frames to be stacked together

    # --- evaluation related ---
    n_eval_episodes: int = 100
    eval_points: tuple = (
        0.1,
        0.25,
        0.5,
        1.0,
    )  # fractions of train steps at which to evaluate agent

    # --- exploration-exploitation strategy related ---
    epsilon_start: float = 1
    epsilon_end: float = 0.05
    exploration_fraction: float = 0.5  # fraction of training steps to explore

    # --- neural network related ---
    n_hidden_units: int = 512  # output of first linear layer

    # --- optimizer related ---
    learning_rate: float = 1e-4

    # --- replay memory related ---
    capacity: int = 1000

    # --- agent algorithm related ---
    v_min: int = -10
    v_max: int = 10
    n_atoms: int = 51

    # --- statistics related ---
    record_statistics_fraction: float = (
        0.01  # fraction of training steps to record past episodic statistics
    )


class MetricsLogger:
    """Logger that stores various episodic metrics."""

    def __init__(self, seed: int, params: Hyperparameters):
        """Initialize logger."""
        num_stats = int(1 / params.record_statistics_fraction)
        self.seed = seed
        self.params = params
        self.episode_returns = np.empty(num_stats)
        self.episode_lengths = np.empty(num_stats)
        self.evaluation_returns = np.empty(len(params.eval_points))
        self.episode_action_values = np.empty(num_stats)
        self.losses = np.empty(num_stats)
        self.policy_entropy = np.empty(num_stats)

        self.index = 0
        self.eval_index = 0

    def __repr__(self):
        return f"MetricsLogger(seed={self.seed}, params={self.params})"

    def __str__(self):
        return f"Env={self.params.env_name}\tSeed={self.seed}"

    def add(
        self,
        episode_return: float,
        episode_length: float,
        episode_action_value: float,
        entropy: float,
        loss: float,
    ):
        """
        Add episode stats.

        Args:
            episode_return: Mean episodic return of past n_eval episodes.
            episode_length: Mean episodic length of past n_eval episodes.
            episode_action_value: Mean predicted action-value of past n_eval episodes.
            entropy: Mean policy entropy of past n_eval episodes.
            loss: Mean loss of past n_eval episodes.

        Returns:

        """
        self.episode_returns[self.index] = episode_return
        self.episode_lengths[self.index] = episode_length
        self.episode_action_values[self.index] = episode_action_value
        self.policy_entropy[self.index] = entropy
        self.losses[self.index] = loss

        self.index += 1

    def add_evaluation_return(self, mean_eval_return: float):
        """Add mean evaluation return obtained at each evaluation point."""
        self.evaluation_returns[self.eval_index] = mean_eval_return
        self.eval_index += 1


# a namedtuple for the action information of an agent's act method
ActionInfo = namedtuple("ActionInfo", field_names=["action", "action_value", "entropy"])


def to_tensor(
    array: Union[np.ndarray, gymnasium.wrappers.LazyFrames],
    normalize: bool = False,
    new_axis: bool = False,
) -> torch.Tensor:
    """
    Takes any array-like object and turns it into a torch.Tensor on `device`.
    For atari image observations, the normalize parameter can be used to change
    the range from [0,255) to [0,1).

    Args:
        array: An array, which can be states, actions, rewards, etc.
        normalize: Whether to normalize image observations
        new_axis: Whether to add a new axis at the first dimension.

    Returns:
        tensor
    """
    tensor = torch.tensor(np.array(array), device=device)

    if new_axis:
        tensor = tensor.unsqueeze(0)

    if normalize:
        tensor = tensor / 255.0

    return tensor


def set_seed(seed: int):
    """Seeding libraries that use random number generators, for reproducibility purposes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return


# %%
# 2. Environment
# --------------
# If we wanted to create environments, then its class implementation would go here. However, since we are going to
# use `gymnasium` environments, all we need is a function to create them. The below function can be used to create
# an environment, and wrap it with the necessary wrappers, such `AtariPreprocessing` for atari environments. The
# `FrameStack` wrapper is used for all environments, even though it is not necessary for environments like Cart Pole,
# however, in these cases only one observation is used. For some atari environments, an agent can receive a
# termination signal once a life is lost, however this is not recommended (Machado et al. (2018)), therefore here
# termination signal is issued only when a game is over.
#
def create_env(params: Hyperparameters, record_video: bool = False) -> gymnasium.Env:
    """
    Create an environment and apply AtariProcessing wrappers if it is an Atari environment.
    Only environments with discrete action spaces are supported.
    All environments are wrapped by the `FrameStack` wrapper.

    Args:
        params: Hyperparameters namedtuple.
        record_video: Whether this env is used to collect frames for video creation.

    Returns:
        env: A gymnasium environment.

    """
    render_mode = "rgb_array" if record_video else None
    env = gymnasium.make(params.env_name, render_mode=render_mode)
    assert isinstance(
        env.action_space, gymnasium.spaces.Discrete
    ), "Only envs with discrete actions-space allowed."

    if params.image_obs:
        env = gymnasium.wrappers.AtariPreprocessing(env=env)

    env = gymnasium.wrappers.FrameStack(env, params.num_frame_stacking)
    env = gymnasium.wrappers.RecordEpisodeStatistics(
        env, deque_size=params.n_eval_episodes
    )

    return env


# %%
# 3. Agent, network and memory
# ----------------------------
# We need to ask what an agent is and what should it be able to do. An agent in this case is an entity that has a
# *memory* to store experiences and remember them for learning, a *network* that perceives observations and provides
# predictions, an *optimizer* which is used for updating the network parameters, that's updating the prediction
# ability. An agent should also be able to ``act`` on observations, as well as have the ability to ``learn`` and
# improve itself.
#
# The agent memory is in itself an object that has mechanisms for storing experiences and overwriting them once
# they become old, as well as providing a way to sample experiences randomly. The agent network defines the structure
# of the neural network and a way to make predictions. The below classes implement exactly these things, that is an
# ``Agent``, together with a ``ReplayMemory`` and a ``Network`` class. Because C51 uses two networks, the agent has
# a *main* and a *target* network, and a way to ``update`` the target network's parameters to be equal to the
# main network's.
# Lastly, since an agent needs to do exploration using epsilon-greedy, there is a way for updating the exploration rate,
# i.e. ``updating epsilon``.
#
class ReplayMemory:
    """Implements a circular replay memory object based on a deque."""

    def __init__(self, params):
        """
        Initialize the replay memory.
        Args:
            params: Hyperparameters
        """
        self._params = params
        self._buffer = collections.deque([], maxlen=params.capacity)

    def push(
        self,
        obs: gymnasium.wrappers.LazyFrames,
        action: int,
        reward: gymnasium.core.SupportsFloat,
        next_obs: gymnasium.wrappers.LazyFrames,
        done: bool,
    ):
        """
        Add a transition to the replay memory. When the buffer is full,
        the oldest transitions are replaced with new ones.

        Args:
            obs: Agent's observation
            action: Executed action.
            reward: Reward received.
            next_obs: Resulting observation.
            done: Terminal state.
        """
        self._buffer.append((obs, action, reward, next_obs, int(done)))

    def sample(self) -> tuple:
        """
        Sample a minibatch of transitions.

        Raises:
             ValueError: if not enough transitions exist to sample.

         Returns:
             5-tuple of obs, actions, rewards, next_obs, dones

        """
        if len(self._buffer) < self._params.batch_size:
            raise ValueError("Not enough transitions to sample a minibatch")

        sample = random.sample(self._buffer, self._params.batch_size)
        return tuple(zip(*sample))


class Network(nn.Module):
    """
    Class implementation of the Deep-Q-Network architecture, where
    it outputs return distributions instead of action-values.
    """

    def __init__(self, params):
        """Initialize the network. Expects hyperparameters object."""
        super().__init__()
        self._params = params

        if self._params.image_obs:
            self._convolutional = nn.Sequential(
                nn.Conv2d(4, 32, (8, 8), (4, 4), 0),
                nn.ReLU(),
                nn.Conv2d(32, 64, (4, 4), (2, 2), 0),
                nn.ReLU(),
                nn.Conv2d(64, 64, (3, 3), (1, 1), 0),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self._convolutional = nn.Sequential()

        in_features = self._output_size()
        out_features = self._params.n_actions * self._params.n_atoms
        n_hidden_units = self._params.n_hidden_units

        self._head = torch.nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden_units, out_features=out_features),
        )

    def _output_size(self) -> int:
        """Compute size of input to first linear layer."""
        with torch.no_grad():
            example_obs = torch.zeros(1, *self._params.obs_shape)
            return int(np.prod(self._convolutional(example_obs).size()))

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        value_dists = self._head(self._convolutional(*args, **kwargs))
        return value_dists.view(
            -1, self._params.n_actions, self._params.n_atoms
        ).softmax(2)


class Agent:
    """
    Class for the Categorical-DQN (C51) agent.
    In essence, for each action, a value distribution is returned by the network,
    from which a statistic such as the mean is computed to get the action-value.
    """

    def __init__(self, params: Hyperparameters):
        """
        Initialize the agent class.

        Args:
            params: A Hyperparameters instance containing the hyperparameters.
        """
        self._params = params

        self._epsilon = params.epsilon_start
        self._epsilon_decay = (params.epsilon_start - params.epsilon_end) / (
            params.exploration_fraction * params.training_steps
        )

        self._delta = (params.v_max - params.v_min) / (params.n_atoms - 1)
        self._z = torch.linspace(params.v_min, params.v_max, params.n_atoms).to(device)

        self.replay_memory = ReplayMemory(params)

        self._main_network = Network(params).to(device)
        self._target_network = Network(params).to(device)
        self.update_target_network()

        self._optimizer = torch.optim.Adam(
            params=self._main_network.parameters(),
            lr=params.learning_rate,
            eps=0.01 / params.batch_size,
        )

    def act(self, state: torch.Tensor) -> ActionInfo:
        """
        Sampling action for a given state. Actions are sampled randomly during exploration.
        The action-value is the max expected value of the action value-distribution.

        Args:
            state: Current state of agent.

        Returns:
            action_info: Information namedtuple about the sampled action.
        """

        with torch.no_grad():
            value_dists = self._main_network(state)
            expected_returns = (self._z * value_dists).sum(2)

        if random.random() > self._epsilon:
            action = expected_returns.argmax(1)
            action_probs = expected_returns.softmax(0)
        else:
            action = torch.randint(high=self._params.n_actions, size=(1,))
            action_probs = torch.ones(self._params.n_actions) / self._params.n_actions

        action_value = expected_returns[0, action].item()
        policy_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum().item()

        action_info = ActionInfo(
            action=action.item(),
            action_value=round(action_value, 2),
            entropy=round(policy_entropy, 2),
        )

        return action_info

    def decrease_epsilon(self):
        if self._epsilon > self._params.epsilon_end:
            self._epsilon -= self._epsilon_decay

    def update_target_network(self):
        """Updating the parameters of the target network to equal the main network's parameters."""
        self._target_network.load_state_dict(self._main_network.state_dict())

    def learn(self) -> float:
        """Learning step, updates the main network through backpropagation. Returns loss."""
        obs, actions, rewards, next_obs, dones = self.replay_memory.sample()

        states = to_tensor(array=obs, normalize=self._params.image_obs)
        actions = to_tensor(array=actions).view(-1, 1).long()
        rewards = to_tensor(array=rewards).view(-1, 1)
        next_states = to_tensor(array=next_obs, normalize=self._params.image_obs)
        dones = to_tensor(array=dones).view(-1, 1)

        # agent predictions
        value_dists = self._main_network(states)
        # gather probs for selected actions
        probs = value_dists[torch.arange(self._params.batch_size), actions.view(-1), :]

        # ------------------------------ Categorical algorithm ------------------------------
        #
        # Since we are dealing with value distributions and not value functions,
        # we can't minimize the loss using MSE(reward+gamma*Q_i-1 - Q_i). Instead,
        # we project the support of the target predictions T_hat*Z_i-1 onto the support
        # of the agent predictions Z_i, and minimize the cross-entropy term of
        # KL-divergence `KL(projected_T_hat*Z_i-1 || Z_i)`.
        #
        with torch.no_grad():
            # target agent predictions
            target_value_dists = self._target_network(next_states)
            target_expected_returns = (self._z * target_value_dists).sum(2)
            target_actions = target_expected_returns.argmax(1)
            target_probs = target_value_dists[
                torch.arange(self._params.batch_size), target_actions, :
            ]

            m = torch.zeros(self._params.batch_size * self._params.n_atoms).to(device)

            Tz = (rewards + (1 - dones) * self._params.gamma * self._z).clip(
                self._params.v_min, self._params.v_max
            )
            bj = (Tz - self._params.v_min) / self._delta

            l, u = torch.floor(bj).long(), torch.ceil(bj).long()

            offset = (
                torch.linspace(
                    0,
                    (self._params.batch_size - 1) * self._params.n_atoms,
                    self._params.batch_size,
                )
                .long()
                .unsqueeze(1)
                .expand(self._params.batch_size, self._params.n_atoms)
                .to(device)
            )

            m.index_add_(
                0,
                (l + offset).view(-1),
                (target_probs * (u + (l == u).long() - bj)).view(-1).float(),
            )
            m.index_add_(
                0, (u + offset).view(-1), (target_probs * (bj - l)).view(-1).float()
            )

            m = m.view(self._params.batch_size, self._params.n_atoms)
        # -----------------------------------------------------------------------------------

        loss = (-((m * torch.log(probs + 1e-8)).sum(dim=1))).mean()

        self._optimizer.zero_grad()  # set all gradients to zero
        loss.backward()  # backpropagate loss through the network
        self._optimizer.step()  # update weights

        return round(loss.item(), 2)


# %%
# 4. Training and evaluation
# --------------------------
# In order to train an agent on a specific environment, we need a function to implement the interaction between the
# two. The ``train`` function below does that, where it creates an agent and an environment based on provided
# hyperparameters, creates buffers for storing intermediate results, and runs the training process. It also stores
# the results into a ``MetricsLogger`` object periodically, and once training has finished this logger is returned.
# We also need to evaluate the agent periodically, and a standard proposed by Machado et al. (2018), an agent should
# be evaluated at different stages during the training where the evaluation is simply the average episodic returns of
# the past ``k`` episodes. In this case an agent is evaluated at 10%, 25%, 50% resp. 100% of the training steps, each
# time taking the average of the past 100 episodes returns.
# Additionally, the ``train`` function allows the agent performance can be recorded at each evaluation point as it plays
# some episodes apart from the training.
# An improvement to this function is to log results to a professional visualization platform instead of manually
# logging and visualizing progress. A recommended such platform is Weights & Biases (wandb.ai).
# Apart from the `train` function, there also another function that can be used to train multiple agents in parallel,
# and a function for letting a random agent play a number of episodes, so that its results can be used for comparison
# with trained agents.
#
def train(
    seed: int, params: Hyperparameters, verbose: bool, record_video: bool = False
) -> MetricsLogger:
    """
    Create agent and environment, and let the agent interact with the environment
    during a number of steps. Collect and return training metrics.

    Args:
        seed: For reproducibility.
        params: Hyperparameters instance.
        verbose: Whether to print training progress periodically.
        record_video: Whether to make a video at each evaluation point.

    Returns:
        results_buffer: Collected statistics of the agent training.
    """
    print(f"Agent with seed {seed} started.")
    start_time = time.perf_counter()
    set_seed(seed)

    steps = 0  # global time steps for the whole training

    # --- Keeping track of some statistics that can explain agent behaviour ---
    episodes_action_values_deque = collections.deque(maxlen=params.n_eval_episodes)
    episodes_policy_entropy_deque = collections.deque(maxlen=params.n_eval_episodes)
    episodes_losses_deque = collections.deque(maxlen=params.n_eval_episodes)
    record_stats_frequency = int(
        params.record_statistics_fraction * params.training_steps
    )
    # fractions of training steps at which an evaluation is done
    evaluation_points = [int(p * params.training_steps) for p in params.eval_points]
    results_buffer = MetricsLogger(seed=seed, params=params)

    frames_list = (
        []
    )  # list that may contain a list of frames to be used for video creation

    env = create_env(params)
    params.n_actions = env.action_space.n
    params.obs_shape = env.observation_space.shape

    agent = Agent(params=params)

    while steps < params.training_steps:
        # --- Start en episode ---
        done = False
        obs, info = env.reset(seed=seed + steps)

        action_value_sum = 0
        policy_entropy_sum = 0
        loss_sum = 0

        # --- Play an episode ---
        while not done:
            action_info = agent.act(to_tensor(obs, params.image_obs, params.image_obs))
            action = action_info.action
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.replay_memory.push(obs, action, reward, next_obs, terminated)
            agent.decrease_epsilon()

            obs = next_obs
            done = terminated or truncated
            steps += 1

            action_value_sum += action_info.action_value
            policy_entropy_sum += action_info.entropy

            if done:
                episode_length = info["episode"]["l"]
                episodes_action_values_deque.append(action_value_sum / episode_length)
                episodes_policy_entropy_deque.append(
                    policy_entropy_sum / episode_length
                )
                if loss_sum > 0:
                    episodes_losses_deque.append(loss_sum / episode_length)

            # train agent periodically
            if steps % params.update_frequency == 0 and steps >= params.learning_starts:
                loss = agent.learn()
                loss_sum += loss

            # Update the target network periodically.
            if (
                steps % params.target_update_frequency == 0
                and steps >= params.learning_starts
            ):
                agent.update_target_network()

            # Record statistics pf past episodes.
            if steps % record_stats_frequency == 0 and steps <= params.training_steps:
                mean_return = np.mean(env.return_queue).round(2)
                mean_length = np.mean(env.length_queue).round()
                mean_action_value = np.mean(episodes_action_values_deque).round(2)
                mean_entropy = np.mean(episodes_policy_entropy_deque).round(2)
                mean_loss = (
                    np.nan
                    if len(episodes_losses_deque) == 0
                    else np.mean(episodes_losses_deque).round(2)
                )
                results_buffer.add(
                    mean_return, mean_length, mean_action_value, mean_entropy, mean_loss
                )

                # print stats if verbose=True
                if verbose:
                    print(
                        f"step:{steps: <10} "
                        f"mean_episode_return={mean_return: <7.2f}  "
                        f"mean_episode_length={mean_length}",
                        flush=True,
                    )

            # evaluate agent
            if steps in evaluation_points:
                mean_eval_return = np.mean(env.return_queue).round(2)
                results_buffer.add_evaluation_return(mean_eval_return)
                if record_video:
                    # create material for an evaluation video
                    frames = collect_video_frames(agent, seed, params)
                    frames_list.append(frames)

    if record_video:
        # create evaluation gif
        video_name = (
            f"../../_static/videos/tutorials/drl_{params.env_name.split('-')[0]}"
        )
        video_path = f"{video_name}"
        create_gif(frames_list, video_path)

    env.close()
    print(f"seed={seed}: runtime={round(time.perf_counter() - start_time, 2)}s")

    return results_buffer


def random_agent_play(params: Hyperparameters) -> np.ndarray:
    """
    Implement a random agent representing baseline performance.
    Return episode rewards, where the number of episodes equals the number
    of times statistics are recorded for the real agent.
    """
    seed = 1
    set_seed(seed)
    steps = 0  # global time steps for the whole training
    n_episodes = int(1 / params.record_statistics_fraction)
    env = create_env(params)

    for episode in range(n_episodes):
        # --- Start en episode ---
        done = False
        _, info = env.reset(seed=seed)

        # --- Play an episode ---
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

            steps += 1

    episode_returns = np.array(env.return_queue)
    env.close()

    return episode_returns


def parallel_training(
    seeds: list,
    params: Hyperparameters,
    verboses: List[bool],
    record_videos: List[bool],
) -> List[MetricsLogger]:
    """
    Train multiple agents in parallel using different seeds for each,
    and return their respective collected results.
    ** Note: this way of parallel training does not work on GPUs.

    Args:
        seeds: A list of seeds for each agent.
        params: The hyperparameters tuple.
        verboses: Whether to print the progress of each agent.
        record_videos: Whether to record evaluation of each agent.


    Returns:
        results: A list containing the collected results of each agent.
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(seeds)) as executor:
        futures_list = [
            executor.submit(train, seed, params, verbose, record_video)
            for seed, verbose, record_video in zip(seeds, verboses, record_videos)
        ]

    return [f.result() for f in concurrent.futures.as_completed(futures_list)]


# %%
# 5. Visualization
# ----------------
# It is important to track an agent's progress while it trains in order to draw conclusions about its learning and
# debug when it's not learning, and to decide what hyperparameters to tweak for better learning.
# This article (https://neptune.ai/blog/reinforcement-learning-agents-training-debug) lists a number of metrics to
# log for a good insight into the agent's learning. In this tutorial a number of them are implemented. At the end of
# training, a plot is made with the following average episodic metrics:
#
# reward and length: to determine how well the agent does on each episode and how long it plays and whether it
# learns to live longer (ex. Cart Pole) or reach the goal state quickly (ex. Lunar Lander),
#
# predicted action-value of the selected actions: this can be compared to the actual episode rewards, and they should
# be similar because that is what the agent predicts to determine how an episode ends, and behaves based on it,
#
# loss: in this case the cross-entropy term of the KL divergence,
#
# policy entropy: which can be used to determine if the exploration phase is enough. If entropy is always high,
# then the agent is unsure about the value of a state, and if it drops rapidly, then it is falsely
# very sure about the value of a state. Ideally, as the agent explores, entropy should be high
# because of the randomness of action-selection, but as the agent exploits more, entropy should
# decrease indicating determinism of predicted state value,
#
# evaluation: lastly, the agent evaluation points is plotted. Note that the evaluation values are identical to
# those of the training rewards, due to the fact the past *k* (``usually k=100``) episode rewards are
# averaged for both metrics.
#
# Also, since agents are trained with different seeds, the average and standard deviation is what is plotted.
#
# The function ``visualize_performance`` does the plotting, while the function ``aggregate_results``
# computes mean&stddev of the different agents results and ``preprocess_results`` does the combining of the metrics.
# Furthermore, the ``collect_video_frames`` is used to play a number of episodes and return the frames, so that
# ``create_gif`` can be used to create a gif of these frames.
#
def aggregate_results(lst: List[np.ndarray]) -> (np.ndarray, np.ndarray):
    """Aggregate a list of arrays to compute their mean and stddev."""
    average = np.mean(lst, axis=0).round(2)
    stddev = np.std(lst, axis=0).round(2)

    return average, stddev


def preprocess_results(
    results: List[MetricsLogger],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Combine data for various metrics and aggregate them across agents. Return the processed data."""
    stats = [
        [res_buffer.episode_returns for res_buffer in results],
        [res_buffer.episode_lengths for res_buffer in results],
        [res_buffer.episode_action_values for res_buffer in results],
        [res_buffer.losses for res_buffer in results],
        [res_buffer.policy_entropy for res_buffer in results],
        [res_buffer.evaluation_returns for res_buffer in results],
    ]

    aggregated_data = [aggregate_results(lst) for lst in stats]

    return aggregated_data


def visualize_performance(
    processed_data: List[Tuple[np.ndarray, np.ndarray]],
    baseline_returns: np.ndarray,
    params: Hyperparameters,
):
    """
    Visualize the aggregated metrics collected by the agents.

    Args:
        processed_data: A list containing tuples of (mean, stddev) for each metric.
        baseline_returns: Array of the random baseline episodic returns.
        params: Hyperparameters namedtuple.

    Returns:

    """
    plt.style.use("seaborn")
    x = np.linspace(
        params.record_statistics_fraction * params.training_steps,
        params.training_steps,
        int(1 / params.record_statistics_fraction),
    )
    color = "royalblue"
    y_labels = [
        "Return",
        "Episode Length",
        "Predicted action-value",
        "Loss",
        "Entropy",
    ]
    titles = [
        "Aggregated agents returns vs baseline",
        "Aggregated episode lengths",
        "Aggregated action-value per episode",
        "Aggregated training losses",
        "Aggregated policy entropy",
    ]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 21))

    eval_axes = axes.flatten()[-1]
    axes = axes.flatten()[:-1]

    for i, ax in enumerate(axes):
        mean, std = processed_data[i]
        ax.plot(x, mean, color=color, label="mean")
        ax.fill_between(
            x=x, y1=mean - std, y2=mean + std, label="stddev", alpha=0.2, color="tomato"
        )
        ax.set(xlabel="Steps", ylabel=y_labels[i], title=titles[i])
        ax.legend()

    axes[0].plot(
        x,
        baseline_returns,
        color="black",
        label="baseline",
    )
    axes[0].legend()

    eval_mean, eval_std = processed_data[-1]
    eval_points = [int(p * params.training_steps) for p in params.eval_points]
    eval_axes.errorbar(
        eval_points,
        eval_mean,
        yerr=eval_std,
        fmt="o-",
        capsize=7,
        label="Mean ± StdDev",
        color=color,
    )
    eval_axes.set(
        xlabel="Step",
        ylabel="Return",
        title="Aggregated evaluation returns at specific steps",
    )
    plt.ylim(min(eval_mean) - max(eval_std) - 25, max(eval_mean) + max(eval_std) + 25)
    eval_axes.legend()

    figname = f"drl_{params.env_name.split('-')[0]}"
    fig.savefig(f"../../_static/img/tutorials/{figname}.png")
    plt.show()

    return


def collect_video_frames(
    agent: Agent, seed: int, params: Hyperparameters, n_episodes: int = 2
) -> list:
    """
    Let agent play a number of episodes and collect list of rendered frames.

    Args:
        agent: A trained agent.
        seed: Integer seed for reproducibility.
        params: Hyperparameters namedtuple.
        n_episodes: Number of episodes to evaluate agent.

    Returns:
        frames: List of rendered frames.

    """
    frames = []
    env = create_env(params, True)

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            frame = env.render()
            frames.append(frame)

            action_info = agent.act(to_tensor(obs, params.image_obs, params.image_obs))
            next_obs, reward, terminated, truncated, info = env.step(action_info.action)
            obs = next_obs
            done = terminated or truncated

    print(f"Evaluation return while recording: {np.mean(env.return_queue):.2f}")
    env.close()

    return frames


def create_gif(frames_list: List[List[np.ndarray]], save_path: str):
    gifs = [
        moviepy.editor.ImageSequenceClip(frames, fps=48).margin(20)
        for frames in frames_list
    ]
    final_gif = moviepy.editor.clips_array([gifs])
    final_gif.write_gif(f"{save_path}.gif")

    return


# %%
# Now we are ready to start training, but first we need to create hyperparameters instances. Below are two different
# sets of hyperparameters, one that is common for atari environments, and the other works decently for easier to learn
# environments.
# Note that the call to the ``parallel_training()`` function must be wrapped within a main block.
#
Easy_Envs_Params = Hyperparameters(
    training_steps=int(5e5),
    learning_starts=int(2e4),
    batch_size=64,
    exploration_fraction=0.3,
    learning_rate=1e-3,
    capacity=int(2e5),
    v_min=-100,
    v_max=100,
    n_atoms=101,
)


Atari_Params = Hyperparameters(
    training_steps=int(6e6),
    learning_starts=int(8e4),
    image_obs=True,
    target_update_frequency=int(1e4),
    num_frame_stacking=4,
    exploration_fraction=0.25,
    learning_rate=2.5e-4,
    capacity=int(1e6),
)


if __name__ == "__main__":
    env_name = "PongNoFrameskip-v4"
    hparams = Atari_Params  # or Easy_Envs_Params
    hparams.env_name = env_name

    if env_name == "CartPole-v1":
        hparams.n_eval_episodes = 500

    agent_seeds = [6, 28, 496, 8128]  # perfect numbers
    verboses = [True, False, False, False]
    record_videos = [True, False, False, False]

    parallel_results = parallel_training(agent_seeds, hparams, verboses, record_videos)

    agent_stats = preprocess_results(parallel_results)
    random_agent_baseline = random_agent_play(params=hparams)

    visualize_performance(agent_stats, random_agent_baseline, hparams)

# %%
# 6. Results
# ----------
# Four agents are trained in parallel on six different environments, and the results are shown below.
# The blue line depicts the mean of the four agents results, while the red shaded area is the stddev.
# The evaluation plot consists of a straight line depicting the mean, and the bars stand for the stddev.
# The gifs consist of four parts, each made at each evaluation point lasting for two episodes.
# Gifs are made for the first seed only.
#
# Acrobot-v1
# ^^^^^^^^^^
# .. image:: /_static/img/tutorials/drl_Acrobot.png
# .. image:: /_static/videos/tutorials/drl_Acrobot.gif
#
# These results suggest that the learning is robust across seeds, and the task is solved efficiently.
#
# Cartpole-v1
# ^^^^^^^^^^^
# .. image:: /_static/img/tutorials/drl_CartPole.png
# .. image:: /_static/videos/tutorials/drl_CartPole.gif
#
# Although the task is solved, the performance degrades somewhat for some seeds. The CartPole-v1 is considered
# solved when an agent scores at least 485 on average for 500 consecutive episodes.
#
# LunarLander-v2
# ^^^^^^^^^^^^^^
# .. image:: /_static/img/tutorials/drl_LunarLander.png
# .. image:: /_static/videos/tutorials/drl_LunarLander.gif
#
# The task is solved here too, despite quite some variance in performance between the seeds.
# Interestingly, the length metric first increases while agents are learning, but then it reverses,
# as agents want to reach the goal state quickly.
#
# BreakoutNoFrameskip-v4
# ^^^^^^^^^^^^^^^^^^^^^^
# .. image:: /_static/img/tutorials/drl_BreakoutNoFrameskip.png
# .. image:: /_static/videos/tutorials/drl_BreakoutNoFrameskip.gif
#
# For Breakout, judging from the plots, it seems like more training would have been beneficial.
# However, comparing it to the reported results for DQN in Machado et al. (2018), these results
# are acceptable.
#
# CrazyClimberNoFrameskip-v4
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# .. image:: /_static/img/tutorials/drl_CrazyClimberNoFrameskip.png
# .. image:: /_static/videos/tutorials/drl_CrazyClimberNoFrameskip.gif
#
# The Crazy Climber environment seems to have been solved rather good, although better than this
# is reported in Bellemare et al. (2017).
#
# PongNoFrameskip-v4
# ^^^^^^^^^^^^^^^^^^
# .. image:: /_static/img/tutorials/drl_PongNoFrameskip.png
# .. image:: /_static/videos/tutorials/drl_PongNoFrameskip.gif
#
# And lastly, training for Pong also looks alright, despite not being solved fully.
# Here too, the length metric reverses, as agents need to beat the opponent fast.

# print("End!")

# %%
#
# This leads ut to the end of this tutorial. Hopefully, it serves as a good place to start coding for deep RL.
# Now you can pick your favorite environment and throw some agents at it, let see how well they do!
# Enjoy coding!
#

# %%
# References
# ----------
# Bellemare et al. (2017): "A Distributional Perspective on Reinforcement Learning"
#
# Machado et al. (2018): "Revisiting the Arcade Learning Environment: Evaluation Protocols and
# Open Problems for General Agents"
#
# =========================================== END OF FILE ===========================================
