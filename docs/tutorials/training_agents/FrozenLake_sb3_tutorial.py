"""
FrozenLake with Stable-Baselines3
=================================

"""

# %%
# In this tutorial, we'll explore how to train an agent on the
# `FrozenLake <https://gymnasium.farama.org/environments/toy_text/frozen_lake/>`__
# environment using the Stable-Baselines3 library, which provides reliable implementations
# of popular reinforcement learning algorithms.
#
# We'll be using the A2C (Advantage Actor-Critic) algorithm, which is a policy-based method
# that uses a value function to reduce the variance of policy gradient updates.
#

# %%
# Now let's import the libraries we'll need for our experiment.
#

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from stable_baselines3 import A2C  # We'll use the Advantage Actor-Critic algorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

# For extracting probabilities from the policy network
from stable_baselines3.common.utils import obs_as_tensor

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


# %%
# Environment Setup
# ----------------
#

# %%
# We'll set parameters to configure our FrozenLake environment. These parameters
# control the size of the grid world and its properties.
#

# Set the parameters needed to generate the FrozenLake environment
map_size = 7  # Size of the square grid
seed = 123  # Random seed for reproducibility
is_slippery = (
    False  # If True, the agent will have a chance to slip to a different direction
)
proba_frozen = 0.9  # Probability that a cell is frozen (walkable) vs a hole

# %%
# Now we create the FrozenLake environment with our specified parameters.
# The environment consists of a grid where the agent must navigate from the start
# position to the goal while avoiding holes in the ice.
#

# Create the FrozenLake environment
env = gym.make(
    "FrozenLake-v1",
    is_slippery=is_slippery,
    render_mode="rgb_array",  # For visualization purposes
    desc=generate_random_map(
        size=map_size, p=proba_frozen, seed=seed  # Generate a random map layout
    ),
)

# %%
# Setting Up Training with Callbacks
# ---------------------------------
#

# %%
# In reinforcement learning, we often don't know exactly how many timesteps
# are needed to train a good model. Instead of setting a fixed number, we can
# use callbacks to stop training when the model reaches a certain performance level.
#

# We don't know how long it takes to train the model, but we know when it's good enough:
# If each run out of a batch of 10 succeeds we stop. We construct this by defining a reward threshold
# and use the eval callback in order to check the threshold after each batch.
# Stop training when the model reaches the reward threshold.
# For more thresholds have a look at:
# https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html#:~:text=A%20callback%20is%20a%20set%20of%20functions%20that,monitoring%2C%20auto%20saving%2C%20model%20manipulation%2C%20progress%20bars%2C%20%E2%80%A6
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.999, verbose=1)
eval_callback = EvalCallback(
    env, callback_on_new_best=callback_on_best, verbose=1, n_eval_episodes=10
)

# %%
# Model Creation and Training
# --------------------------
#

# %%
# Now we'll create and train our A2C model. A2C is an actor-critic algorithm
# that maintains both a policy (actor) and a value function (critic) network.
#

# Create the A2C model and specifiy the MlpPolicy
# A2C stands for the Advantage Actor-Critic algorithm
# For more information have a look at the documentation:
# https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
model = A2C(
    "MlpPolicy", env, verbose=1
)  # MlpPolicy uses neural networks for both actor and critic
model.learn(total_timesteps=20_000, callback=eval_callback, log_interval=100)

# %%
# Testing the Trained Model
# ------------------------
#

# %%
# After training, let's see how the model performs in the environment.
# We'll run the agent until it completes an episode and render the final state.
#

# Get the environment after we finished training
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(map_size * 50):  # Set a reasonable upper limit on steps
    # Get the best action according to the trained policy
    action, _state = model.predict(obs, deterministic=True)
    # Take the action in the environment
    obs, reward, done, info = vec_env.step(action)
    # If the episode is done (reached goal or fell in hole), render and exit
    if done[0]:
        vec_env.render("human")
        break

# %%
# Extracting Policy Information
# ---------------------------
#

# %%
# One advantage of using deep RL libraries like Stable-Baselines3 is that
# we can extract useful information from the trained models. Here, we'll
# extract the action probabilities for each state to understand what our agent learned.
#

# Calculate the learned probabilities for each state-action pair
proba_table = np.ndarray((env.observation_space.n, env.action_space.n))
for i in range(env.observation_space.n):
    # For each state, evaluate the policy
    _, log_prob, _ = model.policy.evaluate_actions(
        # Convert state index to tensor format expected by the policy
        obs_as_tensor(np.array([i], dtype=np.float64), model.policy.device),
        # Test all possible actions (0-3) for this state
        torch.tensor([i for i in list(range(4))]),
    )
    # Convert log probabilities back to normal probabilities and store
    proba_table[i] = np.exp(log_prob.detach().numpy())

# %%
# Visualization Helpers
# -------------------
#

# %%
# To better understand what our agent has learned, we'll create visualization
# functions to display the policy in an intuitive format.
#


# Helper function to convert probability table to directional arrows
def proba_directions_map(proba_table, map_size):
    """Get the best learned action & map it to arrows."""
    # Reshape and find maximum probability values for each state
    proba_table_val_max = proba_table.max(axis=1).reshape(map_size, map_size)
    # Find which action has the highest probability for each state
    proba_table_best_action = np.argmax(proba_table, axis=1).reshape(map_size, map_size)
    # Map action indices to arrow directions for visualization
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    proba_table_directions = np.empty(
        proba_table_best_action.flatten().shape, dtype=str
    )

    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(proba_table_best_action.flatten()):
        if np.abs(proba_table_val_max.flatten())[idx] > eps:
            # Only show arrows where the agent has a clear preference
            # This avoids displaying misleading arrows in states the agent never visited
            # or where all actions have equal probability
            proba_table_directions[idx] = directions[val]
    # Reshape back to grid format for visualization
    proba_table_directions = proba_table_directions.reshape(map_size, map_size)
    return proba_table_val_max, proba_table_directions


def plot_q_values_map(proba_table, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = proba_directions_map(proba_table, map_size)

    # Create a figure with two side-by-side plots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # Left: Display the final state of the environment
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Right: Visualize the learned policy as a heatmap with directional arrows
    sns.heatmap(
        qtable_val_max,  # Color intensity represents probability strength
        annot=qtable_directions,  # Show arrows for the best action
        fmt="",  # No specific format for annotations
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,  # Grid line width
        linecolor="black",
        xticklabels=[],  # No tick labels
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},  # Make arrows larger
    ).set(title="Learned Probabilities\nArrows represent best action")
    # Add a border to the heatmap for clarity
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    plt.show()


# %%
# Visualize the Learned Policy
# --------------------------
#

# %%
# Now we can visualize what our agent has learned. The plot will show the
# final state of the environment and a heatmap of the learned policy.
#

plot_q_values_map(proba_table, env, map_size=map_size)

# %%
# From here on we'll just help to understand how the training progress works.


# In order to display the learning progress we need to create another callback
class CustomLoggingCallback(BaseCallback):
    def __init__(self, interval: int = 100, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.interval = interval
        self.num_episodes = 0

    def _on_step(self) -> bool:
        # Get the rewards for the current step
        reward = self.locals["rewards"]
        done = self.locals["dones"]

        if done[0] == True:
            self.rewards.append(float(reward[0]))
        return True


# %%
# Now that we have defined our custom callback that will track rewards,
# we can instantiate it to use during our training process.
#

# Instantiate the custom callback
logging_callback = CustomLoggingCallback()

# Create a new A2C model
model = A2C("MlpPolicy", env, verbose=1)

# %%
# Train the model with our custom callback to log the rewards.
# The callback will store the rewards at the end of each episode, allowing us
# to visualize the learning progress over time.
#

# Train the model with the custom callback
model.learn(total_timesteps=10_000, callback=[logging_callback], log_interval=100)
# Get the range of episodes for plotting
idx = range(len(logging_callback.rewards))

# %%
# To visualize the learning process effectively, we'll compute a rolling average
# of the rewards. This helps smooth out the noise in the reward signal and
# shows the general trend of improvement.
#


def rolling_mean(x, window_size):
    """Calculate the rolling mean of the given data with specified window size."""
    # Create a window of ones with equal weights (1/window_size for each element)
    window = np.ones(window_size) / window_size
    # Use convolution to efficiently calculate the moving average
    # 'valid' mode means we only return output where the window fully overlaps with input
    return np.convolve(x, window, mode="valid")


# %%
# Now we'll compute a rolling average with a window of 20 episodes.
# This will smooth out short-term fluctuations and highlight the longer-term trend,
# making it easier to see if the agent is consistently improving over time.
#

# Calculate the rolling mean of rewards with a window of 20 episodes
rolling_rewards = rolling_mean(logging_callback.rewards, 20)
# Plot the results to visualize learning progress
plt.plot(range(len(rolling_rewards)), rolling_rewards, label="Reward")
plt.xlabel("Episodes")
plt.ylabel("Average Reward (over 20 episodes)")
plt.title("Learning Progress: How rewards improve during training")
plt.grid(True, alpha=0.3)
plt.legend()
