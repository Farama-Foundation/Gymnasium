"""Gymnasium v26 new step function - Solving Reacher-v4 with REINFORCE.

====================================================================

This tutorial aims to familiarise users with Gymnasium v26's new .step() function
"""

# %%
# Introduction
# ------------------------
# This tutorial aims to familiarise users with Gymnasium v26’s new
# .step() function.
#    In v21, the type definition of step() is
#    tuple\ ``[ObsType, SupportsFloat, bool, dict[str, Any]`` representing
#    the next observation, the reward from the step, if the episode is
#    done, and additional info from the step. Due to reproducibility
#    issues that will be expanded on in a blog post soon, we have changed
#    the type definition to
#    tuple\ ``[ObsType, SupportsFloat, bool, bool, dict[str, Any]``]
#    adding an extra boolean value. This additional bool corresponds to
#    the older done, which has been changed to terminated and
#    truncated.These changes were introduced in Gym v26 (turned off by
#    default in v25).
# In this tutorial, we will be using the REINFORCE algorithm [1] to solve
# the task of 'Reacher'. Reacher is a two-jointed robot arm. The goal is
# to move the robot's end effector (called a fingertip) close to a target
# that is spawned at a random position.

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import pandas as pd
import seaborn as sns
import random

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,5)


# %%
# Policy Network
# ------------------------
# A policy is a mapping from the current environment observation to a probability distribution of the actions to be taken. The policy used in tutorial is 
# parameterized by a neural network. It consists of a 2 linear layers that is shared between both the predicted mean and standard deviation. 
# Further, single individual linear layer is used to estimate the mean and the standard deviation. TanH is used as a non-linearity between the hidden layers.
        
class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, observation_space, action_space) -> None:
        """Initializes a neural network that estimates the mean and standard deviation of a normal distribution from which an action is sampled from.
        
        Args:
        observation_space : int - Dimension of the observation space
        action_space : int - Dimension of the action space
        
        """
        super(Policy_Network, self).__init__()

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
        """Returns the mean and standard deviation of a normal distribution from which an action is sampled from.
        
        Args:
            x : tensor - Observation from the environment
        
        Returns:
            action_means : tensor - predicted mean of the normal distribution
            action_stddevs: tensor - predicted standard deviation of the normal distribution
        
        """
        x = x.float()

        shared_features = self.shared_net(x)

        action_means = self.policy_mean_net(shared_features)
        
        action_stddevs = self.policy_stddev_net(shared_features)
        action_stddevs = torch.log(1 + torch.exp(action_stddevs)) # Kind a activation function log(1 + exp(x))

        return action_means, action_stddevs


# %%
# REINFORCE Algorithm
# ------------------------
# REINFORCE is a policy gradient algorithm that aims to maximize the expected reward. It is a Monte-Carlo variant of policy gradients.
# It involves collecting trajectories using the current policy to update the policy parameters at end of the episode.
# This implementation uses entropy regularization which promotes action diversity

class REINFORCE():
    """REINFORCE algorithm."""
    def __init__(self, observation_space, action_space, beta):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1] to solve the task at hand (Reacher-v4).
        
        Args:
            observation_space : int - Dimension of the observation space
            action_space : int - Dimension of the action space
            beta : float - entropy weight factor (controls exploration)
        
        """
        self.learning_rate = 1e-4 # Learning rate for policy optimization
        self.gamma = 0.99 # Discount factor
        self.beta = beta # Entropy weight - controls the level of exploration

        self.eps = 1e-6 # small number for mathematical stability

        self.probs = [] # Stores probability values of the sampled action
        self.rewards = [] # Stores the corresponding rewards

        self.entropy = 0
        
        self.net = Policy_Network(observation_space, action_space)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr = self.learning_rate)

    def sample_action(self, state):
        """Returns an action, conditioned on the policy and observation.
        
        Args:
            state : array - Observation from the environment
        
        Returns:
            action : float - Action to be performed
        
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample().numpy()
        prob = distrib.log_prob(action) 

        # Calculate and store entropy for future entropy-based exploration
        self.entropy += distrib.entropy().mean()
        self.probs.append(prob)

        return action


    def update(self):
        """Updates the policy network's weights."""
        Running_G = 0
        Gs = []

        # Discounted return (backwards)
        for R in self.rewards[::-1]:
            Running_G = R + self.gamma * Running_G
            Gs.insert(0, Running_G)

        deltas = torch.tensor(Gs)

        loss = 0
        # minimize -1 * prob * (reward obtained  entropy)
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * (delta + (self.beta * self.entropy)) * (-1)
        
        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        self.entropy = 0



# %%
# Train the Agent
# -------------------
# 
# Following is the overview of the training procedure
# 
# 
#    for seed in random seeds:
#        reinitialize agent
#        for episode in max no of episodes:
#            untill episode is done:
#                sample action based on current observation
#                take action to receive reward and next observation
#                store action take, its probability, and the observed reward
#            update the policy
# 

# Create and wrap the environment
env = gym.make("Reacher-v4")
wrapped_env =  gym.wrappers.RecordEpisodeStatistics(env, 50)

episodes = int(7e3) # Total number of episodes
observation_space = 11 # Observation-space of Reacher-v4
action_space = 2 # Action-space of Reacher-v4
reinforce_rewards_across_seeds = []

for seed in [1, 2, 3, 5, 8, 13]: # Fibonacci seeds
    
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(observation_space, action_space, 0.01)

    reinforce_rewards_across_episodes = []

    for e in range(episodes):

        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed = seed)

        done = False

        while not done:

            action = agent.sample_action(obs)
            
            # gymnasium v2g step function returns - tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
            # next observation, the reward from the step, if the episode is terminated, if the episode is truncated and additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            # truncated: The episode duration reaches max number of timesteps
            # terminated: Any of the state space values is no longer finite.

            # done is either of the former ones happening
            done = terminated or truncated

            agent.rewards.append(reward)
        
        reinforce_rewards_across_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if e % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print('Episode:', e, 'Average Reward:', avg_reward)
        
    reinforce_rewards_across_seeds.append(reinforce_rewards_across_episodes)
    print('-' * 20)



# %%
# Plot learning curve
# -------------------
# 

rewards_to_plot = [[reward[0] for reward in rewards] for rewards in reinforce_rewards_across_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns = {'variable':'episodes', 'value':'reward'}, inplace = True)
sns.set(style='darkgrid', context='talk', palette='rainbow')
sns.lineplot(x="episodes", y="reward", data=df1).set(title = 'REINFORCE for Reacher-v4')
plt.show()




# %%
# References
# -------------------
# 
# [1] Williams, Ronald J.. “Simple statistical gradient-following
# algorithms for connectionist reinforcement learning.” Machine Learning 8
# (2004): 229-256.
# 
