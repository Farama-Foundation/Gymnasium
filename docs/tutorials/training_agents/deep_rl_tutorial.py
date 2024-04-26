"""
Implementation of a Deep RL Agent
=================================

intro: what is this tutorial and why. what will it contain

In this tutorial we describe and show how a deep reinforcement learning agent is implemented and trained using pytorch
v2.0.1. The agent is C51. But there is not much difference between different agents, they all are implemented similarly.

Let's start by importing necessary libraries:
"""

# Global TODOs:
# TODO: train agent on Lunar-Lander env.
# TODO: savefig
# TODO: Final check on documentation and typing.


# %%
__author__ = "Hardy Hasan"
__date__ = "2023-04-18"
__license__ = "MIT License"

import concurrent.futures
import copy
import functools
import random
from collections import namedtuple
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
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
class ResultsBuffer:
    """
    This class stores various metrics from episodes/steps of agent-environment interaction.
    """

    __slots__ = [
        "index",
        "seed",
        "params",
        "episode_returns",
        "episode_lengths",
        "training_eval_returns",
        "evaluation_returns",
        "episode_action_values",
        "losses",
        "exploration",
        "policy_entropy",
    ]

    def __init__(self, seed: int, params: namedtuple):
        self.seed = seed
        self.params = params
        self.episode_returns = np.zeros(params.training_steps)
        self.episode_lengths = np.zeros(params.training_steps)
        self.training_eval_returns = np.zeros(
            int(1 / params.evaluation_frequency_percentage)
        )
        self.evaluation_returns = None
        self.episode_action_values = np.zeros(params.training_steps)
        self.losses = np.zeros(params.training_steps)
        self.exploration = np.zeros(params.training_steps)
        self.policy_entropy = np.zeros(params.training_steps)

        self.index = 0

    def __repr__(self):
        return f"ResultsBuffer(seed={self.seed}, params={self.params})"

    def __str__(self):
        return f"Env={self.params.env_name}\tSeed={self.seed}"

    def add(
        self,
        episode_return: float,
        episode_length: float,
        episode_action_value: float,
        loss: float,
        epsilon: float,
        entropy: float,
    ) -> None:
        """
        Add step stats.
        Args:
            episode_return: Latest episodic return
            episode_length: Latest episode length
            episode_action_value: Predicted average action-value of the last episode.
            loss: Latest training loss
            epsilon: Current exploration value.
            entropy: Current policy entropy.

        Returns:

        """
        if self.index < self.params.training_steps:
            self.episode_returns[self.index] = episode_return
            self.episode_lengths[self.index] = episode_length
            self.episode_action_values[self.index] = episode_action_value
            self.losses[self.index] = loss
            self.exploration[self.index] = epsilon
            self.policy_entropy[self.index] = entropy

        self.index += 1

    def add_training_eval(self, eval_return: float):
        """Add evaluation return obtained during training."""
        eval_freq = (
            self.params.training_steps * self.params.evaluation_frequency_percentage
        )
        ind = int(self.index // eval_freq - 1)
        self.training_eval_returns[ind] = eval_return

    def add_evaluation_returns(self, eval_returns: list):
        """Add evaluation returns obtained after training."""
        self.evaluation_returns = np.array(eval_returns)


# a namedtuple for the experience of an agent at each training step
Experience = namedtuple(
    "Experience", field_names=["obs", "action", "next_obs", "reward", "done"]
)


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

    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)

    if normalize:
        return tensor / 255.0

    return tensor


def set_seed(seed: int):
    """Setting seeding for libraries that use random number generators, for reproducibility purposes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return


def create_atari_env(env: gymnasium.Env, params: namedtuple) -> gymnasium.Env:
    """
    Creates an atari environment and applies AtariProcessing and FrameStack wrappers.

    Args:
        env: A gymnasium atari environment. Assumes no frame-skipping is done.
        params: Hyperparameters namedtuple.

    Returns:

    """
    env = gymnasium.wrappers.AtariPreprocessing(env=env)
    env = gymnasium.wrappers.FrameStack(
        env=env, num_stack=params.num_frame_stacking, lz4_compress=True
    )

    return env


def parallel_training(
    seeds: list, params: namedtuple, verbose: bool = False
) -> List[ResultsBuffer]:
    """
    Train multiple agents in parallel using different seeds for each,
    and return their respective collected results.

    Args:
        seeds: A list of seeds for each agent.
        params: The hyperparameters tuple.
        verbose: Whether to print the progress of each agent asynchronously.


    Returns:
        results: A list containing the collected results of each agent.
    """
    partial = functools.partial(train, params=params, verbose=verbose)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(partial, seeds)

    results = [res for res in results]

    return results


def exponential_moving_average(
    data_points: np.ndarray, alpha: float = 0.25
) -> np.ndarray:
    """Computing the exponential moving average of an array of data points."""
    assert 0 <= alpha < 1, "Smoothing factor alpha is out of range"
    exp_moving_average = np.empty(len(data_points))

    # Find the first non -inf index
    first_non_neg_index = np.argmax(data_points != -np.inf)

    # Handle the case when the first element is -inf
    if first_non_neg_index > 0:
        exp_moving_average[:first_non_neg_index] = data_points[:first_non_neg_index]

    exp_moving_average[first_non_neg_index] = data_points[first_non_neg_index]

    for i in range(first_non_neg_index + 1, len(data_points)):
        exp_moving_average[i] = (
            alpha * data_points[i] + (1 - alpha) * exp_moving_average[i - 1]
        )

    return exp_moving_average.round(2)


def aggregate_results(lst: List[np.ndarray]) -> (np.ndarray, np.ndarray):
    """Aggregate a list of arrays to provide their mean and stddev."""
    average = np.mean(lst, axis=0).round(2)
    stddev = np.std(lst, axis=0).round(2)

    return average, stddev


def preprocess_results(
    results: List[ResultsBuffer], alpha: float = 0.25
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Smooth data for various metrics and aggregate them across agents. Return the processed data."""
    stats = [
        [res_buffer.episode_returns for res_buffer in results],
        [res_buffer.episode_lengths for res_buffer in results],
        [res_buffer.episode_action_values for res_buffer in results],
        [res_buffer.losses for res_buffer in results],
        [res_buffer.exploration for res_buffer in results],
        [res_buffer.policy_entropy for res_buffer in results],
    ]

    smoothed_data = [
        [exponential_moving_average(lst, alpha) for lst in stat] for stat in stats
    ]

    eval_data = [
        [res_buffer.training_eval_returns for res_buffer in results],
        [res_buffer.evaluation_returns for res_buffer in results],
    ]

    return [aggregate_results(lst) for lst in smoothed_data + eval_data]


def visualize_performance(
    processed_data: List[Tuple[np.ndarray, np.ndarray]],
    baseline_return: np.ndarray,
    params: namedtuple,
):
    """
    Visualize the aggregated metrics collected by the agents.

    Args:
        processed_data: A list containing tuples of (mean, stddev) for each metric.
        baseline_return: Array of the baseline return.
        params: Hyperparameters namedtuple.

    Returns:

    """
    plt.style.use("seaborn-v0_8-darkgrid")

    color = "purple"
    y_labels = [
        "Return",
        "Episode Length",
        "Predicted action-value",
        "Loss",
        "Epsilon",
        "Entropy",
        "Return",
        "Return",
    ]
    titles = [
        "Aggregated agents returns vs baseline",
        "Aggregated episode lengths",
        "Aggregated action-value per episode",
        "Aggregated training losses",
        "Aggregated epsilon decay",
        "Aggregated policy entropy",
        "Aggregated training evaluation returns",
        "Aggregated evaluation returns",
    ]

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 28))

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        mean, std = processed_data[i]
        x = range(1, len(mean) + 1)
        ax.plot(x, mean, color=color, label="mean")
        ax.fill_between(
            x=x, y1=mean - std, y2=mean + std, label="stddev", alpha=0.2, color="grey"
        )
        ax.set(xlabel="Steps", ylabel=y_labels[i], title=titles[i])
        ax.legend()

    axes[0].plot(
        range(1, len(baseline_return) + 1),
        baseline_return,
        color="black",
        label="baseline",
    )
    axes[0].legend()

    # figname = f"drl_{params.env_name.split('-')[0]}"
    # fig.savefig(f"../../_static/img/tutorials/{figname}.png")
    plt.show()

    return None


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
    upfront whether enough memory is available on the machine, instead of the training
    being quit unexpectedly.
    Each term in an experience (state, action , next_state, reward ,done) is stored into
    separate arrays. A maximum capacity is required upon initialization, as well as
    observation shape, and whether the observation is image data, such as in Atari games.

    In Atari games, the frame stacking technique is used, where the past four observations make
    up a state. Thus, for each experience, `state` and `next_state` are four frames each, however
    the first three frames of `next_state` is the last three frames of `state`, hence these frames
    are stored once in the `next_states` array, and when sampling, concatenated back to build a
    proper `next_state`.
    """

    def __init__(self, params: namedtuple) -> None:
        """
        Initialize a replay memory.

        Args:
            params: A namedtuple containing all hyperparameters needed for training an agent,
                    hence it contains all the parameters needed for creating a memory buffer.
        """
        self.params = params
        self.length: int = 0  # number of experiences stored so far
        self.index: int = 0  # current index to store data to

        # shape of obs buffers differ depending on whether an obs is image data.
        if self.params.image_obs:
            self._obs_shape = (
                self.params.capacity,
                self.params.num_frame_stacking,
                *self.params.obs_shape,
            )
            self._next_obs_shape = (
                self.params.capacity,
                1,
                *self.params.obs_shape,
            )
        else:
            self._obs_shape = (self.params.capacity, *self.params.obs_shape)
            self._next_obs_shape = (self.params.capacity, *self.params.obs_shape)

        self._obs: np.ndarray = np.zeros(self._obs_shape, np.float16)
        self._actions: np.ndarray = np.zeros(self.params.capacity, np.uint8)
        self._next_obs: np.ndarray = np.zeros(self._next_obs_shape, np.float16)
        self._rewards: np.ndarray = np.zeros(self.params.capacity, np.float16)
        self._dones: np.ndarray = np.zeros(self.params.capacity, np.uint8)

    def __len__(self):
        """Returns the number of experiences stored in the memory."""
        return self.length

    def push(self, experience: Experience) -> None:
        """
        Adds a new experience into the buffer

        Args:
            experience: step experience of agent.

        Returns:

        """
        self._obs[self.index] = experience.obs
        self._actions[self.index] = experience.action
        self._next_obs[self.index] = experience.next_obs
        self._rewards[self.index] = experience.reward
        self._dones[self.index] = experience.done

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

        obs = self._obs[indices]
        next_obs = self._next_obs[indices]
        if self.params.image_obs:
            next_obs = np.concatenate((obs[:, 1:, :, :], next_obs), axis=1)

        actions = self._actions[indices]
        rewards = self._rewards[indices]
        dones = self._dones[indices]

        if self.params.image_obs:
            assert np.equal(
                obs[:, 1:, :], next_obs[:, :3, :]
            ), "Incorrect concatenation."

        return obs, actions, next_obs, rewards, dones


class Agent(nn.Module):
    """
    Class for agent running on Categorical-DQN (C51) algorithm.
    In essence, for each action, a value distribution is returned,
    from which a statistic such as the mean is computed to get the
    action-value.
    """

    def __init__(self, params: namedtuple):
        """
        Initializing the agent class.

        Args:
            params: A namedtuple containing the hyperparameters.
        """
        super().__init__()
        self.params = params
        self.epsilon = self.params.epsilon_start
        self.eps_reduction = (self.params.epsilon_start - self.params.epsilon_end) / (
            self.params.anneal_length_percentage * self.params.training_steps
        )

        self.delta = (self.params.v_max - self.params.v_min) / (self.params.n_atoms - 1)

        self.replay_memory = ReplayMemory(self.params)

        # metrics
        self.loss = float("-inf")
        self.predicted_action_value = float("-inf")
        self.selected_action_probs = float("-inf")
        self.policy_entropy = float("-inf")

        # The support is the set of values over which a probability
        # distribution is defined and has non-zero probability there.
        self.support = torch.linspace(
            start=self.params.v_min, end=self.params.v_max, steps=self.params.n_atoms
        ).to(device)

        # -- defining the neural network --
        in_features = self.params.in_features
        out_features = self.params.n_actions * self.params.n_atoms
        n_hidden_units = self.params.n_hidden_units

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

        self.head = torch.nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden_units, out_features=out_features),
        )

        self.optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.params.learning_rate
        )

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
        value_dist = self.convolutional(self.head(state))

        return value_dist.view(state.shape[0], self.params.n_actions, -1).softmax(2)

    def act(self, state: torch.Tensor, exploit: bool) -> Union[int, torch.Tensor]:
        """
        Sampling action for a given state. Actions are sampled randomly during exploration.
        The action-value is the max expected value of the action value-distribution.

        Args:
            state: Current state of agent.
            exploit: True when not exploring.

        Returns:
            action: The sampled action.
        """
        random_value = random.random()

        with torch.no_grad():
            value_dist = self.forward(state)
            expected_returns = torch.sum(self.support * value_dist, dim=2)

        if exploit or random_value > self.epsilon:
            action = torch.argmax(expected_returns, dim=1)
            action_probs = expected_returns.softmax(0)
        else:
            action = torch.randint(high=self.params.n_actions, size=(1,), device=device)
            action_probs = torch.ones(self.params.n_actions) / self.params.n_actions

        self.selected_action_probs = value_dist[torch.arange(state.shape[0]), action, :]

        if len(action) == 1:
            action = action.item()
            self.predicted_action_value = (
                (self.support * self.selected_action_probs).sum().item()
            )
            self.policy_entropy = -(action_probs * torch.log(action_probs)).sum().item()

        return action

    def decrease_epsilon(self):
        if self.epsilon > self.params.epsilon_end:
            self.epsilon -= self.eps_reduction

    def get_metrics(self) -> Tuple[float, float, float, float]:
        """
        Return metrics such as loss, action-value,
        policy entropy, exploration rate epsilon at each step.
        """
        return self.loss, self.predicted_action_value, self.epsilon, self.policy_entropy

    def learn(self, target_agent: "Agent"):
        """
        Learning steps, which includes updating the network parameters through backpropagation.
        Args:
            target_agent: The target agent that is used as an oracle.

        """
        obs, actions, next_obs, rewards, dones = self.replay_memory.sample()

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
        _ = target_agent.act(next_states, exploit=True)
        target_probs = target_agent.selected_action_probs

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
            torch.linspace(
                start=0,
                end=(self.params.batch_size - 1) * self.params.n_atoms,
                steps=self.params.batch_size,
            )
            .long()
            .unsqueeze(1)
            .expand(self.params.batch_size, self.params.n_atoms)
            .to(device)
        )

        m.index_add_(
            0,
            (l + offset).view(-1),
            (target_probs * (u + (l == u).long() - bj)).view(-1),
        )
        m.index_add_(0, (u + offset).view(-1), (target_probs * (bj - l)).view(-1))

        m = m.view(self.params.batch_size, self.params.n_atoms)
        # -----------------------------------------------------------------------------------

        loss = (-((m * torch.log(probs)).sum(dim=1))).mean()

        # set all gradients to zero
        self.optimizer.zero_grad()

        # backpropagate loss through the network
        loss.backward()

        # update weights
        self.optimizer.step()

        self.loss = round(loss.item(), 2)

        return


# %%
# Training
# --------
# Describe how training is performed, what is needed, mention that hyperparameters will be explained later.
# Describe useful statistics to plot during training to track agent progress.


def train(seed: int, params: namedtuple, verbose: bool) -> ResultsBuffer:
    """
    Creates agent and environment, and lets the agent interact
    with the environment until it learns a good policy.

    Args:
        seed: For reproducibility.
        params: A namedtuple containing all necessary hyperparameters.
        verbose: Whether to print training progress periodically.

    Returns:
        results_buffer: An instance of the ResultsBuffer class.
    """
    set_seed(seed)

    steps = 0  # global time steps for the whole training
    evaluate_frequency = params.evaluation_frequency_percentage * params.training_steps
    episode_return = float("-inf")
    episode_length = float("-inf")
    episode_action_value = float("-inf")
    results_buffer = ResultsBuffer(seed=seed, params=params)

    env = gymnasium.make(params.env_name)
    assert isinstance(
        env.action_space, gymnasium.spaces.Discrete
    ), "Only envs with discrete actions-space allowed."
    if params.image_obs:
        env = create_atari_env(env=env, params=params)

    env = gymnasium.wrappers.RecordEpisodeStatistics(env)
    evaluate_env = env

    agent = Agent(params=params).to(device)
    target_agent = copy.deepcopy(agent).to(device)
    # target parameters are frozen.
    for p in target_agent.parameters():
        p.requires_grad = False

    while steps < params.training_steps:
        # --- Start en episode ---
        done = False
        obs, info = env.reset(seed=seed)
        action_value_sum = 0

        # --- Play an episode ---
        while not done:
            state = to_tensor(obs, params.image_obs)
            action = agent.act(state=state, exploit=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            step_experience = Experience(obs, action, next_obs, reward, terminated)
            agent.replay_memory.push(step_experience)
            agent.decrease_epsilon()

            obs = next_obs
            done = terminated or truncated
            steps += 1

            loss, action_value, epsilon, entropy = agent.get_metrics()
            action_value_sum += action_value

            if done:
                episode_return, episode_length = (
                    info["episode"]["r"],
                    info["episode"]["l"],
                )
                episode_action_value = action_value_sum / info["episode"]["l"]

            results_buffer.add(
                episode_return,
                episode_length,
                episode_action_value,
                loss,
                epsilon,
                entropy,
            )

            # train agent periodically if enough experience exists
            if steps % params.update_frequency == 0 and steps >= params.learning_starts:
                agent.learn(target_agent)

            # Update the target network periodically.
            if (
                steps % params.target_update_frequency == 0
                and steps >= params.learning_starts
            ):
                target_agent.load_state_dict(agent.state_dict())

            # evaluate agent periodically
            if steps % evaluate_frequency == 0:
                eval_return = evaluate(agent, evaluate_env, params.image_obs, seed)
                results_buffer.add_training_eval(eval_return)

            # print progress periodically
            if verbose and steps % 10_000 == 0:
                mean_episode_return = np.mean(env.return_queue).round()
                mean_episode_length = np.mean(env.length_queue).round()
                print(
                    f"step:{steps:<10} mean_episode_return:{mean_episode_return:<7} "
                    f"mean_episode_length:{mean_episode_length}",
                    flush=True,
                )

            """CODE FOR DEBUGGING GRADIENTS"""
            # if steps % 1000 == 0:
            #     grads = [param.grad for name, param in agent.named_parameters() if param.grad is not None]
            #     avg_grad = [tens.mean() for tens in grads if isinstance(tens, torch.Tensor)]
            #     print(avg_grad)

    # --- Evaluate agent on five different seeds than training seed once training has finished
    evaluation_returns = [
        evaluate(agent, evaluate_env, params.image_obs, seed + i) for i in range(1, 6)
    ]
    results_buffer.add_evaluation_returns(evaluation_returns)

    # TODO: clean up, create results summary, return stuff
    env.close()
    evaluate_env.close()

    return results_buffer


def random_agent_play(params: namedtuple) -> np.ndarray:
    """Implement a random agent play representing baseline performance. Return episode rewards."""
    seed = 1
    set_seed(seed)
    steps = 0  # global time steps for the whole training
    env = gymnasium.wrappers.RecordEpisodeStatistics(gymnasium.make(params.env_name))
    returns = np.zeros(params.training_steps)
    eps_return = float("-inf")

    while steps < params.training_steps:
        # --- Start en episode ---
        done = False
        _, info = env.reset(seed=seed)

        # --- Play an episode ---
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated

            if done:
                eps_return = info["episode"]["r"]

            if steps < params.training_steps:
                returns[steps] = eps_return

            steps += 1

    env.close()

    return returns


# %%
# Evaluation
# ----------
# Describe how an agent should be evaluated once its training is finished.


def evaluate(
    agent: Agent, env: gymnasium.Env, image_obs: bool, seed: int, n_episodes: int = 20
) -> float:
    """
    Evaluate an agent on an environment for a number of
    episodes and return the average episode reward.

    Args:
        agent: A trained agent.
        env: Environment to evaluate on.
        image_obs: Whether observations are image arrays.
        seed: Integer seed for reproducibility.
        n_episodes: Number of episodes to evaluate agent.

    Returns:
        mean_return: Average episode return.

    """
    eval_return = 0

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            state = to_tensor(obs, image_obs)
            action = agent.act(state, exploit=True)
            next_obs, reward, terminated, truncated, info = env.step(action)

            obs = next_obs
            done = terminated or truncated

            if done:
                eval_return += info["episode"]["r"]

    return round(eval_return / n_episodes, 2)


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
        "learning_starts",  # number of steps taken before agent does any learning.
        "obs_shape",  # a tuple representing shape of observations, ex. (1, 4), (4, 84)
        "image_obs",  # boolean, indicating whether the env provides image observations
        "batch_size",  # number of experiences to sample for updating agent network parameters
        "update_frequency",  # how often to update agent network parameters
        "target_update_frequency",  # how often to replace target agent network parameters
        "gamma",  # discount factor
        "num_frame_stacking",  # number of frames to be stacked together
        # --- training related ---
        "evaluation_frequency_percentage",  # evaluating agent, as percentage of training steps
        # --- exploration-exploitation strategy related ---
        "epsilon_start",
        "epsilon_end",
        "anneal_length_percentage",  # percentage of training steps to decrease eps during training
        # --- neural network related ---
        "in_features",  # number of features/parameters passed to the first linear layer
        "n_hidden_units",  # output of first linear layer
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

# train for different seeds.
# evaluate all agents.
# plot the progress and evaluation.

LunarLander_hyperparameters = Hyperparameters(
    env_name="LunarLander-v3",
    n_actions=4,
    training_steps=int(5e5),
    learning_starts=1e4,
    obs_shape=(1, 8),
    image_obs=False,
    batch_size=32,
    update_frequency=4,
    target_update_frequency=200,
    gamma=0.99,
    num_frame_stacking=0,
    evaluation_frequency_percentage=0.20,
    epsilon_start=1,
    epsilon_end=0.05,
    anneal_length_percentage=0.30,
    in_features=8,
    n_hidden_units=256,
    learning_rate=1e-4,
    capacity=int(1e5),
    v_min=-30,
    v_max=30,
    n_atoms=51,
)


CartPole_hyperparameters = Hyperparameters(
    env_name="CartPole-v1",
    n_actions=2,
    training_steps=int(5e4),
    learning_starts=1000,
    obs_shape=(1, 4),
    image_obs=False,
    batch_size=32,
    update_frequency=4,
    target_update_frequency=200,
    gamma=0.99,
    num_frame_stacking=0,
    evaluation_frequency_percentage=0.20,
    epsilon_start=1,
    epsilon_end=0.05,
    anneal_length_percentage=0.40,
    in_features=4,
    n_hidden_units=256,
    learning_rate=1e-4,
    capacity=int(1e5),
    v_min=-100,
    v_max=100,
    n_atoms=101,
)


Acrobot_hyperparameters = Hyperparameters(
    env_name="Acrobot-v1",
    n_actions=3,
    training_steps=int(5e4),
    learning_starts=1000,
    obs_shape=(1, 6),
    image_obs=False,
    batch_size=32,
    update_frequency=4,
    target_update_frequency=200,
    gamma=0.99,
    num_frame_stacking=0,
    evaluation_frequency_percentage=0.20,
    epsilon_start=1,
    epsilon_end=0.05,
    anneal_length_percentage=0.40,
    in_features=6,
    n_hidden_units=256,
    learning_rate=1e-4,
    capacity=int(1e5),
    v_min=-100,
    v_max=100,
    n_atoms=101,
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
    agent_seeds = [11, 13, 17, 19]
    LunarLander_parallel_results = parallel_training(
        seeds=agent_seeds, params=Acrobot_hyperparameters, verbose=True
    )
    random_agent_baseline = random_agent_play(params=Acrobot_hyperparameters)

    smooth_factor_alpha = 0.50
    data = preprocess_results(LunarLander_parallel_results, smooth_factor_alpha)
    random_agent_baseline_smoothed = exponential_moving_average(
        random_agent_baseline, smooth_factor_alpha
    )
    visualize_performance(data, random_agent_baseline_smoothed, Acrobot_hyperparameters)

# %%
# CartPole-v0 visualization
# -------------------------
# .. image:: /_static/img/tutorials/drl_CartPole.png
#
#

# RecordVideo
# how to evaluate a trained agent


# %%
# Finishing words
# ---------------
# whatever is remaining to be said.


# =========================================== END OF FILE ===========================================
