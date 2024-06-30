"""
Implementation of a Deep RL Agent
=================================

intro: what is this tutorial and why. what will it contain

In this tutorial we describe and show how a deep reinforcement learning agent is implemented and trained using pytorch
v2.0.1. The agent is C51. But there is not much difference between different agents, they all are implemented similarly.

Let's start by importing necessary libraries:
"""

# Global TODOs:
# TODO: Final check on documentation and typing.


# %%
__author__ = "Hardy Hasan"
__date__ = "2023-06-30"
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
# %%
# Hyperparameters
# ---------------
# What hyperparameters are necessary, what are common good values etc.
# Describe the importance of seeds.
@dataclasses.dataclass
class Hyperparameters:
    # --- env related ---
    env_name: str = ''
    n_actions: int = 0

    # --- training related ---
    training_steps: int = 1000  # number of steps to train agent for
    learning_starts: int = 100  # number of steps played before agent learns.
    obs_shape: tuple = (1, )  # a tuple representing shape of observations, ex. (1, 4), (4, 84)
    image_obs: bool = False  # boolean, indicating whether the env provides image observations
    batch_size: int = 32  # number of experiences to sample for updating agent network parameters
    update_frequency: int = 4  # how often to update agent network parameters
    target_update_frequency: int = 1000  # how often to replace target agent network parameters
    gamma: float = 0.99  # discount factor
    num_frame_stacking: int = 1  # number of frames to be stacked together

    # --- evaluation related ---
    n_eval_episodes: int = 100
    eval_points: tuple = (0.1, 0.25, 0.5, 1.0)  # fractions of train steps at which to evaluate agent

    # --- exploration-exploitation strategy related ---
    epsilon_start: float = 1
    epsilon_end: float = 0.05
    exploration_fraction: float = 0.5  # fraction of training steps to decrease epsilon during training

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
    record_statistics_fraction: float = 0.01  # fraction of training steps to record past episodic statistics


class ResultsBuffer:
    """
    This class stores various metrics from episodes/steps of agent-environment interaction.
    """

    __slots__ = [
        "index",
        "eval_index",
        "seed",
        "params",
        "episode_returns",
        "episode_lengths",
        "evaluation_returns",
        "episode_action_values",
        "losses",
        "policy_entropy",
    ]

    def __init__(self, seed: int, params: Hyperparameters):
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
        return f"ResultsBuffer(seed={self.seed}, params={self.params})"

    def __str__(self):
        return f"Env={self.params.env_name}\tSeed={self.seed}"

    def add(
        self,
        episode_return: float,
        episode_length: float,
        episode_action_value: float,
        entropy: float,
        loss: float
    ):
        """
        Add step stats.
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
        """Add mean evaluation return obtained after training."""
        self.evaluation_returns[self.eval_index] = mean_eval_return
        self.eval_index += 1


# a namedtuple for the action information of an agent's act method
ActionInfo = namedtuple("ActionInfo", field_names=["action", "action_value", "entropy"])


def to_tensor(array: Union[np.ndarray, gymnasium.wrappers.LazyFrames],
              normalize: bool = False,
              new_axis: bool = False) -> torch.Tensor:
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
    """Setting seeding for libraries that use random number generators, for reproducibility purposes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return


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
    render_mode = 'rgb_array' if record_video else None
    env = gymnasium.make(params.env_name, render_mode=render_mode)
    assert isinstance(
        env.action_space, gymnasium.spaces.Discrete
    ), "Only envs with discrete actions-space allowed."

    if params.image_obs:
        env = gymnasium.wrappers.AtariPreprocessing(env=env)

    env = gymnasium.wrappers.FrameStack(env, params.num_frame_stacking)
    env = gymnasium.wrappers.RecordEpisodeStatistics(env, deque_size=params.n_eval_episodes)

    return env


def parallel_training(
    seeds: list, params: Hyperparameters, verboses: List[bool], record_videos: List[bool]
) -> List[ResultsBuffer]:
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
        futures_list = [executor.submit(train, seed, params, verbose, record_video)
                        for seed, verbose, record_video in zip(seeds, verboses, record_videos)]

    return [f.result() for f in concurrent.futures.as_completed(futures_list)]


def aggregate_results(lst: List[np.ndarray]) -> (np.ndarray, np.ndarray):
    """Aggregate a list of arrays to provide their mean and stddev."""
    average = np.mean(lst, axis=0).round(2)
    stddev = np.std(lst, axis=0).round(2)

    return average, stddev


def preprocess_results(results: List[ResultsBuffer]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Smooth data for various metrics and aggregate them across agents. Return the processed data."""
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
    x = np.linspace(params.record_statistics_fraction*params.training_steps,
                    params.training_steps,
                    int(1/params.record_statistics_fraction))
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

    axes[0].plot(x, baseline_returns, color="black", label="baseline",)
    axes[0].legend()

    eval_mean, eval_std = processed_data[-1]
    eval_points = [int(p * params.training_steps) for p in params.eval_points]
    eval_axes.errorbar(eval_points, eval_mean, yerr=eval_std, fmt='o-', capsize=7, label='Mean ± StdDev', color=color)
    eval_axes.set(xlabel="Step", ylabel="Return", title="Aggregated evaluation returns at specific steps")
    plt.ylim(min(eval_mean) - max(eval_std) - 25, max(eval_mean) + max(eval_std) + 25)
    eval_axes.legend()

    figname = f"drl_{params.env_name.split('-')[0]}"
    fig.savefig(f"../../_static/img/tutorials/{figname}.png")
    plt.show()

    return


# %%
# Agent
# -----
# What is the agent, how is it defined, what can it do, why do we do it as we do?
# Write a short description.


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

    def push(self,
             obs: gymnasium.wrappers.LazyFrames,
             action: int,
             reward: gymnasium.core.SupportsFloat,
             next_obs: gymnasium.wrappers.LazyFrames,
             done: bool):
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

    def sample(self) -> Union[None, tuple]:
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


class Agent(nn.Module):
    """
    Class for agent running on Categorical-DQN (C51) algorithm.
    In essence, for each action, a value distribution is returned,
    from which a statistic such as the mean is computed to get the
    action-value.
    """

    def __init__(self, params: Hyperparameters):
        """
        Initializing the agent class.

        Args:
            params: A namedtuple containing the hyperparameters.
        """
        super(Agent, self).__init__()
        self.params = params
        self.epsilon = self.params.epsilon_start
        self.eps_reduction = (self.params.epsilon_start - self.params.epsilon_end) / (
            self.params.exploration_fraction * self.params.training_steps
        )

        self.delta = (self.params.v_max - self.params.v_min) / (self.params.n_atoms - 1)
        # The support is the set of values over which a probability
        # distribution is defined and has non-zero probability there.
        self.support = torch.linspace(self.params.v_min, self.params.v_max, self.params.n_atoms).to(device)

        self.replay_memory = ReplayMemory(self.params)

        # -- defining the neural network --
        if self.params.image_obs:
            self.convolutional = nn.Sequential(
                nn.Conv2d(4, 32, (8, 8), (4, 4), 0),
                nn.ReLU(),
                nn.Conv2d(32, 64, (4, 4), (2, 2), 0),
                nn.ReLU(),
                nn.Conv2d(64, 64, (3, 3), (1, 1), 0),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.convolutional = nn.Sequential()

        in_features = self._output_size()

        self.head = torch.nn.Sequential(
            nn.Linear(in_features=in_features, out_features=self.params.n_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=self.params.n_hidden_units, out_features=self.params.n_actions * self.params.n_atoms),
        )

        self.optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.params.learning_rate
        )

    def _output_size(self):
        with torch.no_grad():
            example_obs = torch.zeros(1, *self.params.obs_shape)
            return int(np.prod(self.convolutional(example_obs).size()))

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
        value_dist = self.head(self.convolutional(state))

        return value_dist.view(state.shape[0], self.params.n_actions, -1).softmax(2)

    def act(self, state: torch.Tensor) -> ActionInfo:
        """
        Sampling action for a given state. Actions are sampled randomly during exploration.
        The action-value is the max expected value of the action value-distribution.
        During evaluation, actions are exploited with small chance for exploration.

        Args:
            state: Current state of agent.

        Returns:
            action_info: Information namedtuple about the sampled action.
        """
        random_value = random.random()

        with torch.no_grad():
            value_dist = self.forward(state)
            expected_returns = torch.sum(self.support * value_dist, dim=2)

        if random_value > self.epsilon:
            action = torch.argmax(expected_returns, dim=1)
            action_probs = expected_returns.softmax(0)
        else:
            action = torch.randint(high=self.params.n_actions, size=(1,), device=device)
            action_probs = torch.ones(self.params.n_actions) / self.params.n_actions

        action_value = expected_returns[0, action].item()
        policy_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum().item()

        action_info = ActionInfo(action=action.item(),
                                 action_value=round(action_value, 2),
                                 entropy=round(policy_entropy, 2))

        return action_info

    def decrease_epsilon(self):
        if self.epsilon > self.params.epsilon_end:
            self.epsilon -= self.eps_reduction

    def learn(self, target_agent: "Agent") -> float:
        """
        Learning steps, which includes updating the network parameters through backpropagation.
        Args:
            target_agent: The target agent that is used as an oracle.

        Returns:
            loss: The loss value.

        """
        obs, actions, rewards, next_obs, dones = self.replay_memory.sample()

        states = to_tensor(array=obs, normalize=self.params.image_obs).float()
        actions = to_tensor(array=actions).view(-1, 1).long()
        next_states = to_tensor(array=next_obs, normalize=self.params.image_obs).float()
        rewards = to_tensor(array=rewards).view(-1, 1)
        dones = to_tensor(array=dones).view(-1, 1)

        # agent predictions
        value_dists = self.forward(states)
        # gather probs for selected actions
        probs = value_dists[torch.arange(self.params.batch_size), actions.view(-1), :]

        # target agent predictions
        with torch.no_grad():
            target_value_dist = target_agent.forward(next_states)
            target_expected_returns = torch.sum(target_agent.support * target_value_dist, dim=2)
            target_actions = torch.argmax(target_expected_returns, dim=1)
            target_probs = target_value_dist[torch.arange(self.params.batch_size), target_actions, :]

        # ------------------------------ Categorical algorithm ------------------------------
        #
        # Since we are dealing with value distributions and not value functions,
        # we can't minimize the loss using MSE(reward+gamma*Q_i-1 - Q_i). Instead,
        # we project the support of the target predictions T_hat*Z_i-1 onto the support
        # of the agent predictions Z_i, and minimize the cross-entropy term of
        # KL-divergence `KL(projected_T_hat*Z_i-1 || Z_i)`.
        #

        m = torch.zeros(self.params.batch_size * self.params.n_atoms).to(device)

        Tz = (rewards + (1 - dones) * self.params.gamma * self.support).clip(
            self.params.v_min, self.params.v_max
        )
        bj = (Tz - self.params.v_min) / self.delta
        assert (bj >= 0).all() and (
            bj < self.params.n_atoms
        ).all(), "wrong computation for bj"

        l, u = torch.floor(bj).long(), torch.ceil(bj).long()

        # performing a double for-loop, one loop for the minibatch samples and another loop for the atoms, in one step.
        offset = (
            torch.linspace(0, (self.params.batch_size - 1) * self.params.n_atoms, self.params.batch_size)
            .long()
            .unsqueeze(1)
            .expand(self.params.batch_size, self.params.n_atoms)
            .to(device)
        )

        m.index_add_(0,  (l + offset).view(-1), (target_probs * (u + (l == u).long() - bj)).view(-1).to(torch.float32))
        m.index_add_(0, (u + offset).view(-1), (target_probs * (bj - l)).view(-1).to(torch.float32))

        m = m.view(self.params.batch_size, self.params.n_atoms)
        # -----------------------------------------------------------------------------------

        loss = (-((m * torch.log(probs + 1e-8)).sum(dim=1))).mean()

        # set all gradients to zero
        self.optimizer.zero_grad()

        # backpropagate loss through the network
        loss.backward()

        # update weights
        self.optimizer.step()

        return round(loss.item(), 2)


# %%
# Training
# --------
# Describe how training is performed, what is needed, mention that hyperparameters will be explained later.
# Describe useful statistics to plot during training to track agent progress.


def train(seed: int, params: Hyperparameters, verbose: bool, record_video: bool = False) -> ResultsBuffer:
    """
    Create agent and environment, and let the agent interact
    with the environment during a number of steps.

    Args:
        seed: For reproducibility.
        params: A namedtuple containing all necessary hyperparameters.
        verbose: Whether to print training progress periodically.
        record_video: Whether to make an evaluation video.

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
    record_stats_frequency = int(params.record_statistics_fraction * params.training_steps)
    # fractions of training steps at which an evaluation is done
    evaluation_points = [int(p * params.training_steps) for p in params.eval_points]
    results_buffer = ResultsBuffer(seed=seed, params=params)

    frames_list = []  # list that may contain a list of frames to be used for video creation

    env = create_env(params)

    agent = Agent(params=params).to(device)
    target_agent = Agent(params=params).to(device)
    target_agent.load_state_dict(agent.state_dict())

    while steps < params.training_steps:
        # --- Start en episode ---
        done = False
        obs, info = env.reset(seed=seed)

        action_value_sum = 0
        policy_entropy_sum = 0
        loss_sum = 0

        # --- Play an episode ---
        while not done:
            action_info = agent.act(to_tensor(obs, params.image_obs, params.image_obs))
            action = action_info.action
            next_obs, reward, terminated, truncated, info = env.step(action)

            # step_experience = Experience(obs, action, next_obs, reward, terminated)
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
                episodes_policy_entropy_deque.append(policy_entropy_sum / episode_length)
                if loss_sum > 0:
                    episodes_losses_deque.append(loss_sum / episode_length)

            # Record statistics pf past episodes.
            if steps % record_stats_frequency == 0 and steps <= params.training_steps:
                mean_episode_return = np.mean(env.return_queue).round(2)
                mean_episode_length = np.mean(env.length_queue).round()
                mean_action_value = np.mean(episodes_action_values_deque).round(2)
                mean_entropy = np.mean(episodes_policy_entropy_deque).round(2)
                mean_loss = np.nan if len(episodes_losses_deque) == 0 else np.mean(episodes_losses_deque).round(2)

                if verbose:
                    print(f"step:{steps: <10} "
                          f"mean_episode_return={mean_episode_return: <7.2f}  "
                          f"mean_episode_length={mean_episode_length}")

                results_buffer.add(
                    mean_episode_return,
                    mean_episode_length,
                    mean_action_value,
                    mean_entropy,
                    mean_loss
                )

            # train agent periodically if enough experience exists
            if steps % params.update_frequency == 0 and steps >= params.learning_starts:
                loss = agent.learn(target_agent)
                loss_sum += loss

            # Update the target network periodically.
            if steps % params.target_update_frequency == 0 and steps >= params.learning_starts:
                target_agent.load_state_dict(agent.state_dict())

            # evaluate agent
            if steps in evaluation_points:
                mean_eval_return = np.mean(env.return_queue).round(2)
                results_buffer.add_evaluation_return(mean_eval_return)
                if record_video:
                    # create material for an evaluation video
                    frames = collect_video_frames(agent, seed, params)
                    frames_list.append(frames)

    # TODO: clean up, create results summary, return stuff

    if record_video:
        # create evaluation video
        video_name = f"drl_{params.env_name.split('-')[0]}"
        video_path = f"../../_static/videos/tutorials/{video_name}"
        create_gif(frames_list, video_path)

    env.close()
    print(f"Time: {round(time.perf_counter() - start_time, 2)}s")

    return results_buffer


def random_agent_play(params: Hyperparameters) -> np.ndarray:
    """Implement a random agent play representing baseline performance. Return episode rewards."""
    seed = 1
    set_seed(seed)
    steps = 0  # global time steps for the whole training
    n_episodes = int(1/params.record_statistics_fraction)
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


# %%
# Evaluation
# ----------
# Describe how an agent should be evaluated once its training is finished.
def collect_video_frames(agent: Agent, seed: int, params: Hyperparameters, n_episodes: int = 2) -> list:
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

    print(f"Evaluation return: {np.mean(env.return_queue):.2f}")
    env.close()

    return frames


def create_gif(frames_list: List[List[np.ndarray]], save_path: str):
    gifs = [moviepy.editor.ImageSequenceClip(frames, fps=48).margin(20) for frames in frames_list]
    final_gif = moviepy.editor.clips_array([gifs])
    final_gif.write_gif(f"{save_path}.gif")

    return
# %%
# Env1
# ---------------

# train for different seeds.
# evaluate all agents.
# plot the progress and evaluation.


Easy_Envs_Params = Hyperparameters(
    training_steps=int(5e5),
    learning_starts=int(2e4),
    batch_size=64,
    exploration_fraction=0.2,
    learning_rate=1e-3,
    capacity=int(2e5),
    v_min=-100,
    v_max=100,
    n_atoms=101,
)


Atari_Params = Hyperparameters(
    training_steps=int(4e6),
    learning_starts=int(4e4),
    image_obs=True,
    target_update_frequency=2000,
    num_frame_stacking=4,
    exploration_fraction=0.25,
    learning_rate=2.5e-4,
    capacity=int(5e5),
)
# %%
# Env2
# ---------------
# Define hyperparameters dict.
# train for different seeds.
# evaluate all agents.
# plot the progress and evaluation.
# env2_hyperparameters = Hyperparameters()

env2 = None

if __name__ == "__main__":
    # %%
    # LunarLander-v3 training
    # --------------------
    env_name = 'Pong-v4'
    hparams = Atari_Params
    hparams.env_name = env_name

    if env_name == 'CartPole-v1':
        hparams.n_eval_episodes = 500

    agent_seeds = [6, 28, 496, 8128]
    verboses = [True, False, False, False]
    record_videos = [True, False, False, False]

    global_start_time = time.perf_counter()
    parallel_results = parallel_training(agent_seeds[:1], hparams, verboses, record_videos)
    print(f"Global runtime: {round(time.perf_counter()-global_start_time, 2)}s")

    agent_stats = preprocess_results(parallel_results)
    random_agent_baseline = random_agent_play(params=hparams)

    visualize_performance(agent_stats, random_agent_baseline, hparams)

# %%
# CartPole-v0 visualization
# -------------------------
# .. image:: /_static/img/tutorials/drl_CartPole.png
# .. image:: /_static/img/tutorials/drl_Acrobot.png
# .. image:: /_static/img/tutorials/drl_LunarLander.png
# .. image:: /_static/videos/tutorials/drl_CartPole.gif
# .. image:: /_static/videos/tutorials/drl_Acrobot.gif
# .. image:: /_static/videos/tutorials/drl_LunarLander.gif
#
#

# RecordVideo
# how to evaluate a trained agent


# %%
# Finishing words
# ---------------
# whatever is remaining to be said.


# =========================================== END OF FILE ===========================================
