"""
Solving Lunar Lander with Deep Q-Learning (DQN)
===============================================

This tutorial implements Deep Q-Learning (DQN) from scratch with PyTorch and uses it to land the
lunar module in ``LunarLander-v3``.
"""

# %%
# Introduction
# ------------
#
# The tabular Q-learning tutorials (`Blackjack <blackjack_q_learning.html>`_ and
# `FrozenLake <frozenlake_q_learning.html>`_) store one value per state-action pair in a table.
# That works when there are few states, but Lunar Lander's observation is a vector of eight
# continuous numbers (position, velocity, angle, angular velocity and two leg-contact flags), so
# no table could hold every state. DQN [1] replaces the table with a neural network
# :math:`Q_\theta(s, a)` that takes the observation and outputs one estimated return per action.
#
# Two ideas make training that network stable:
#
# - **Experience replay**: transitions are stored in a buffer and the network is trained on random
#   minibatches from it, which breaks the correlation between consecutive steps and lets each
#   transition be reused many times.
# - **A target network**: a slowly updated copy :math:`Q_{\theta^-}` of the network produces the
#   regression targets, so the targets do not shift every time the network is updated.
#
# Each update regresses :math:`Q_\theta(s, a)` towards the one-step target
#
# .. math::
#
#    y = r + \gamma \, (1 - \text{terminated}) \, \max_{a'} Q_{\theta^-}(s', a')
#
# This tutorial pays particular attention to the ``(1 - terminated)`` term, because it is where the
# difference between ``terminated`` and ``truncated`` in ``env.step()`` matters most.
#
# **Lunar Lander** has four discrete actions: do nothing, fire the left orientation engine, fire
# the main engine, and fire the right orientation engine. The reward grows as the lander gets
# closer to the pad, slower and more level, landing safely adds 100 points, crashing costs 100,
# and firing the engines costs a little fuel each frame. An episode scoring 200 or more counts as
# a solution. More information on the environment can be found at
# https://gymnasium.farama.org/environments/box2d/lunar_lander/.
#
# The `A2C tutorial <vector_a2c.html>`_ trains a policy-gradient agent on the same environment, so
# the two can be compared side by side.
#

# Author: Vishwa
# License: MIT License

from __future__ import annotations

import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import gymnasium as gym

# %%
# Terminated versus truncated
# ---------------------------
#
# ``env.step(action)`` returns ``next_obs, reward, terminated, truncated, info``. Both flags end the
# episode, but they mean different things for learning:
#
# - ``terminated`` is ``True`` when the episode reached a terminal state of the task: here the lander
#   crashed, flew off the screen, or came to rest. There is no future reward after a terminal state,
#   so the target is just :math:`y = r`.
# - ``truncated`` is ``True`` when the episode was cut short for a reason outside the task. For
#   ``LunarLander-v3`` this is the ``TimeLimit`` wrapper stopping the episode after 1000 steps.
#   The lander is still hovering and future reward still exists, so the target must keep bootstrapping
#   from :math:`Q_{\theta^-}(s', \cdot)`.
#
# A common bug is to compute ``done = terminated or truncated`` and use ``done`` in the target. That
# teaches the agent that running out of time is the same as a crash. Here we store ``terminated``
# in the replay buffer and only use ``terminated or truncated`` to decide when to reset the
# environment. The `Handling Time Limits <../gymnasium_basics/handling_time_limits.html>`_
# tutorial covers this distinction in more detail.
#
# We can see both flags by running the same random episode twice: once with the default time limit
# of 1000 steps, where the lander crashes long before the limit (``terminated``), and once with a
# time limit of 50 steps, which cuts the episode short while the lander is still in the air
# (``truncated``):

for max_episode_steps in (1000, 50):
    env = gym.make("LunarLander-v3", max_episode_steps=max_episode_steps)
    obs, info = env.reset(seed=0)
    env.action_space.seed(0)
    episode_over, n_steps = False, 0
    while not episode_over:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        n_steps += 1
        episode_over = terminated or truncated
    print(f"{max_episode_steps=}: {terminated=}, {truncated=} after {n_steps} steps")
    env.close()


# %%
# The Q-network and the replay buffer
# -----------------------------------
#
# The Q-network is a small multilayer perceptron mapping the 8-dimensional observation to one
# Q-value per action. The replay buffer is a fixed-size ``deque``: once full, the oldest
# transitions are dropped as new ones arrive.


class QNetwork(nn.Module):
    """Maps an observation to one estimated return (Q-value) per action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        """Builds a two-hidden-layer MLP.

        Args:
            obs_dim: Size of the observation vector
            n_actions: Number of discrete actions
            hidden_dim: Width of each hidden layer
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the Q-values for a batch of observations."""
        return self.net(obs)


class ReplayBuffer:
    """Stores transitions and samples random minibatches from them."""

    def __init__(self, capacity: int):
        """Creates an empty buffer.

        Args:
            capacity: Maximum number of transitions kept
        """
        self.memory = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, terminated):
        """Stores one transition. Only ``terminated`` is stored, not ``truncated``."""
        self.memory.append((obs, action, reward, next_obs, terminated))

    def sample(self, batch_size: int):
        """Returns a random minibatch as PyTorch tensors."""
        batch = random.sample(self.memory, batch_size)
        obs, actions, rewards, next_obs, terminated = zip(*batch, strict=True)
        return (
            torch.as_tensor(np.array(obs), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.int64),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.array(next_obs), dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.float32),
        )

    def __len__(self):
        """Returns the number of stored transitions."""
        return len(self.memory)


# %%
# The DQN agent
# -------------
#
# The agent holds the online network :math:`Q_\theta` and the target network :math:`Q_{\theta^-}`.
#
# - ``act`` chooses actions epsilon-greedily: with probability :math:`\epsilon` a random action
#   (exploration), otherwise the action with the highest Q-value (exploitation).
# - ``learn`` samples a minibatch, computes the target :math:`y` with the ``(1 - terminated)`` mask
#   and takes one gradient step on the mean squared error between :math:`Q_\theta(s, a)` and
#   :math:`y`.
# - The target network then moves a small step ``tau`` towards the online network
#   (a "soft" or Polyak update), so the targets change slowly.
#
# The network is trained every ``learn_every`` environment steps, once the buffer holds at least
# one minibatch.


class DQNAgent:
    """Deep Q-Learning agent with experience replay and a soft-updated target network."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        learning_rate: float = 5e-4,
        discount_factor: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        tau: float = 1e-3,
        learn_every: int = 4,
    ):
        """Initializes the networks, optimizer and replay buffer.

        Args:
            obs_dim: Size of the observation vector
            n_actions: Number of discrete actions
            learning_rate: Adam learning rate
            discount_factor: Discount factor (gamma)
            buffer_size: Replay buffer capacity
            batch_size: Minibatch size per update
            tau: Soft update rate of the target network
            learn_every: Number of environment steps between updates
        """
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.tau = tau
        self.learn_every = learn_every

        self.q_network = QNetwork(obs_dim, n_actions)
        self.target_network = QNetwork(obs_dim, n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.step_count = 0

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Chooses an action epsilon-greedily."""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q_values = self.q_network(torch.as_tensor(obs, dtype=torch.float32))
        return int(q_values.argmax())

    def step(self, obs, action, reward, next_obs, terminated):
        """Stores a transition and learns every ``learn_every`` steps."""
        self.memory.push(obs, action, reward, next_obs, terminated)
        self.step_count += 1
        if (
            self.step_count % self.learn_every == 0
            and len(self.memory) >= self.batch_size
        ):
            self.learn()

    def learn(self):
        """Takes one gradient step on a random minibatch."""
        obs, actions, rewards, next_obs, terminated = self.memory.sample(
            self.batch_size
        )

        with torch.no_grad():
            next_q = self.target_network(next_obs).max(dim=1).values
            # No bootstrapping after a terminal state; truncated transitions still bootstrap
            targets = rewards + self.discount_factor * (1.0 - terminated) * next_q

        q_values = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(
                self.target_network.parameters(),
                self.q_network.parameters(),
                strict=True,
            ):
                target_param.mul_(1.0 - self.tau).add_(self.tau * param)


# %%
# Training
# --------
#
# Epsilon starts at 1 (fully random) and decays by a factor of 0.995 after each episode down to
# 0.01. Training stops once the average score over the last 100 episodes reaches 200, or after
# ``n_episodes`` episodes.
#
# In our tests on a two-core CPU, four different seeds solved the environment after 744 to 1,029
# episodes, taking 5 to 8 minutes. Lower ``n_episodes`` while experimenting.

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = gym.make("LunarLander-v3")
env.action_space.seed(seed)
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

n_episodes = 2000
epsilon, epsilon_end, epsilon_decay = 1.0, 0.01, 0.995
solved_score = 200.0

episode_returns = []
recent_returns = deque(maxlen=100)
for episode in range(n_episodes):
    obs, info = env.reset(seed=seed + episode)
    episode_return = 0.0
    episode_over = False
    while not episode_over:
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.step(obs, action, reward, next_obs, terminated)
        obs = next_obs
        episode_return += reward
        episode_over = terminated or truncated

    episode_returns.append(episode_return)
    recent_returns.append(episode_return)
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    if (episode + 1) % 100 == 0:
        print(
            f"Episode {episode + 1:4d} | average return (last 100): {np.mean(recent_returns):7.1f}"
            f" | epsilon: {epsilon:.3f}"
        )
    if len(recent_returns) == 100 and np.mean(recent_returns) >= solved_score:
        print(f"Solved after {episode + 1} episodes!")
        break
env.close()


# %%
# Visualizing the training
# ------------------------
#

window = min(100, len(episode_returns))
moving_average = np.convolve(episode_returns, np.ones(window) / window, mode="valid")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(episode_returns, alpha=0.3, label="Episode return")
ax.plot(
    np.arange(window - 1, len(episode_returns)),
    moving_average,
    label=f"{window}-episode average",
)
ax.axhline(solved_score, color="k", linestyle="--", linewidth=1, label="Solved (200)")
ax.set_xlabel("Episode")
ax.set_ylabel("Return")
ax.set_title("DQN on LunarLander-v3")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# .. image:: /_static/img/tutorials/dqn_lunar_lander_training.png
#
# The 100-episode average starts around -150 while the agent explores randomly and mostly crashes. It climbs
# as the agent first learns to hover instead of crashing, then to land on the pad and switch the
# engines off.


# %%
# Evaluating the agent
# --------------------
#
# Training scores include exploration noise, so we evaluate the greedy policy (``epsilon=0``) on
# new episodes. Reaching a 100-episode average of 200 during training does not guarantee the same
# average on new episodes: across the four seeds above, the greedy policy averaged 171 to 216, with
# 47% to 76% of episodes scoring 200 or more. Training for longer after the first time the threshold
# is reached usually makes the policy more consistent.

eval_env = gym.make("LunarLander-v3")
eval_returns = []
for episode in range(100):
    obs, info = eval_env.reset(seed=10_000 + episode)
    episode_return = 0.0
    episode_over = False
    while not episode_over:
        obs, reward, terminated, truncated, info = eval_env.step(agent.act(obs))
        episode_return += reward
        episode_over = terminated or truncated
    eval_returns.append(episode_return)
eval_env.close()

eval_returns = np.array(eval_returns)
print(
    f"Greedy policy over 100 episodes: mean {eval_returns.mean():.1f}, "
    f"{(eval_returns >= solved_score).mean():.0%} of episodes scored 200 or more"
)


# %%
# Watching the agent land
# -----------------------
#
# Set ``render_mode="human"`` to watch the trained agent in a window.

# watch_env = gym.make("LunarLander-v3", render_mode="human")
# obs, info = watch_env.reset()
# episode_over = False
# while not episode_over:
#     obs, reward, terminated, truncated, info = watch_env.step(agent.act(obs))
#     episode_over = terminated or truncated
# watch_env.close()


# %%
# Next steps
# ----------
#
# - Try ``LunarLander-v3`` with ``continuous=True``. DQN needs a discrete action space, so a
#   continuous-action method such as SAC or TD3 is needed there.
# - Common DQN extensions are Double DQN [2], which reduces the over-estimation of Q-values by
#   choosing :math:`a'` with the online network and evaluating it with the target network, and
#   prioritized experience replay [3].
# - For tuned, tested implementations, see `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/>`_
#   and `CleanRL <https://docs.cleanrl.dev/>`_.


# %%
# References
# ----------
#
# [1] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland,
# G. Ostrovski, et al. "Human-level control through deep reinforcement learning" Nature (2015).
#
# [2] H. van Hasselt, A. Guez, D. Silver. "Deep Reinforcement Learning with Double Q-learning" AAAI (2016).
#
# [3] T. Schaul, J. Quan, I. Antonoglou, D. Silver. "Prioritized Experience Replay" ICLR (2016).
