"""
Solving Reaper-v4 using REINFORCE
=================================

**This tutorial aims to familiarise users with Gymnasium v26’s new
.step() function.**

   In v21, the type definition of step() is
   tuple\ ``[ObsType, SupportsFloat, bool, dict[str, Any]`` representing
   the next observation, the reward from the step, if the episode is
   done, and additional info from the step. Due to reproducibility
   issues that will be expanded on in a blog post soon, we have changed
   the type definition to
   tuple\ ``[ObsType, SupportsFloat, bool, bool, dict[str, Any]``]
   adding an extra boolean value. This additional bool corresponds to
   the older done, which has been changed to terminated and
   truncated.These changes were introduced in Gym v26 (turned off by
   default in v25).

In this tutorial, we will be using the REINFORCE algorithm [1] to solve
the task of *Reacher*. Reacher is a two-jointed robot arm. The goal is
to move the robot’s end effector (called a fingertip) close to a target
that is spawned at a random position.

"""

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.normal import Normal

import pandas as pd
import seaborn as sns
import random

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams["figure.figsize"] = (10,5)

import warnings
warnings.filterwarnings("ignore")

env = gym.make("Reacher-v4")

class Policy_Network(nn.Module):

    def __init__(self, observation_space, action_space) -> None:
        super(Policy_Network, self).__init__()

        """
        Initializes a neural network that estimates the mean and standard deviation of a normal distribution from which an action is sampled from
        
        Args:
        observation_space : int - Dimension of the observation space
        action_space : int - Dimension of the action space
        """

        hidden_space1 = 16
        hidden_space2 = 32

        # Shared Network 
        self.shared_net = nn.Sequential(nn.Linear(observation_space, hidden_space1),
                                     nn.Tanh(),
                                     nn.Linear(hidden_space1, hidden_space2),
                                     nn.Tanh())
        
        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(nn.Linear(hidden_space2, action_space))

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(nn.Linear(hidden_space2, action_space))

    def forward(self, x):

        """
        Returns the mean and standard deviation of a normal distribution from which an action is sampled from
        
        Args:
        x : tensor - Observation from the environment
        """

        x = x.float()

        shared_features = self.shared_net(x)

        action_means = self.policy_mean_net(shared_features)
        
        action_stddevs = self.policy_stddev_net(shared_features)
        action_stddevs = torch.log(1 + torch.exp(action_stddevs)) # Kind a activation function log(1 + exp(x))

        return action_means, action_stddevs

class REINFORCE():
    def __init__(self, observation_space, action_space, beta):

        """
        Initializes an agent that learns a policy to solve the task at hand (Reacher-v4)
        
        Args:
        observation_space : int - Dimension of the observation space
        action_space : int - Dimension of the action space
        beta : float - entropy weight factor (controls exploration)
        """

        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.beta = beta

        self.eps = 1e-6

        self.probs = []
        self.values = []
        self.rewards = []

        self.entropy = 0
        
        self.net = Policy_Network(observation_space, action_space)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr = self.learning_rate)

    def sample_action(self, state):

        """
        Returns an action, conditioned on the policy and observation
        
        Args:
        state : array - Observation from the environment
        """

        state = torch.tensor([state])
        action_means, action_stddevs = self.net(state)

        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)

        action = distrib.sample()
        prob = distrib.log_prob(action) 

        # Calculate and store entropy for future entropy-based exploration
        self.entropy += distrib.entropy().mean()
        self.probs.append(prob)

        return action.numpy()


    def update(self):

        """
        Updates the policy network's weights
        
        """

        Running_G = 0
        Gs = []

        # Discounted return (backwards)
        for R in self.rewards[::-1]:
            Running_G = R + self.gamma * Running_G
            Gs.insert(0, Running_G)

        deltas = torch.tensor(Gs)

        loss = 0
        # minimize the -1 * prob * (reward obtained  entropy) 
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * (delta + (self.beta * self.entropy)) * (-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # Empty / zero out all episode-centric variables
        self.probs = []
        self.rewards = []
        self.entropy = 0

episodes = int(5e3)

observation_space = 11
action_space = 2

reinforce_rewards_across_seeds = []

for seed in [1, 2, 3, 5, 8, 13]:
    
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    agent = REINFORCE(observation_space, action_space, 0.01)

    reinforce_rewards_across_episodes = []

    for e in range(episodes):

        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = env.reset(seed = seed)

        done = False
        episode_reward = 0

        while not done:

            action = agent.sample_action(obs)
            
            # gymnasium v2g step function returns - tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
            # next observation, the reward from the step, if the episode is terminated, if the episode is truncated and additional info from the step
            obs, reward, terminated, truncated, info = env.step(action)

            # truncated: The episode duration reaches max number of timesteps
            # terminated: Any of the state space values is no longer finite.

            # done is either of the former ones happening
            done = terminated or truncated

            agent.rewards.append(reward)
            episode_reward += reward
        
        reinforce_rewards_across_episodes.append(episode_reward)
        agent.update()

        avg_reward = int(np.average(reinforce_rewards_across_episodes[-50:]))

        if e % 100 == 0:
            print('Episode:', e, 'Average Reward:', avg_reward)
        
    reinforce_rewards_across_seeds.append(reinforce_rewards_across_episodes)
    print('-' * 20)

df1 = pd.DataFrame(reinforce_rewards_across_seeds).melt()
df1.rename(columns = {'variable':'episodes', 'value':'reward'}, inplace = True)
sns.set(style='darkgrid', context='talk', palette='rainbow')
sns.lineplot(x="episodes", y="reward", data=df1).set(title = 'REINFORCE for Reacher-v4')
plt.show()


# %%
# References
# ~~~~~~~~~~
# 
# [1] Williams, Ronald J.. “Simple statistical gradient-following
# algorithms for connectionist reinforcement learning.” Machine Learning 8
# (2004): 229-256.
# 