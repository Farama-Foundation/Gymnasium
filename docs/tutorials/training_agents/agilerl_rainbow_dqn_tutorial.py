"""
AgileRL Rainbow-DQN Implementation
==================================

"""

# %%
# In this tutorial, we will be training a single Rainbow-DQN agent (without HPO) to beat the
# Gymnasium classic control cart-pole environment. AgileRL is a deep reinforcement learning
# library, focussed on improving the RL training process through evolutionary hyperparameter
# optimisation (HPO), which has resulted in upto 10x faster HPO compared to other popular deep RL
# libraries. Check out the AgileRL github `repository <https://github.com/AgileRL/AgileRL/>`__ for
# more information about the library.

# %%
# Rainbow-DQN Overview
# --------------------
# Rainbow DQN is an extension of DQN that integrates multiple improvements and techniques to achieve
# state-of-the-art performance. These improvements include:
#
# * Double DQN (DDQN): Addresses the overestimation bias of Q-values by using two networks to decouple
#   the selection and evaluation of the action in the Q-learning target.
# * Prioritized Experience Replay: Instead of uniformly sampling from the replay buffer, it samples more
#   important transitions more frequently based on the magnitude of their temporal difference (TD) error.
# * Dueling Networks: Splits the Q-network into two separate streams — one for estimating the state value
#   function and another for estimating the advantages for each action. They are then combined to produce
#   Q-values.
# * Multi-step Learning (n-step returns): Instead of using just the immediate reward for learning, it uses
#   multi-step returns which consider a sequence of future rewards.
# * Distributional RL: Instead of estimating the expected value of the cumulative future reward, it predicts
#   the entire distribution of the cumulative future reward.
# * Noisy Nets: Adds noise directly to the weights of the network, providing a way to explore the environment
#   without the need for epsilon-greedy exploration.
# * Categorical DQN (C51): A specific form of distributional RL where the continuous range of possible
#   cumulative future rewards is discretized into a fixed set of categories.
#


# %%
# Dependencies
# ------------
#

import os

import imageio
import numpy as np
import torch
from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
)
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train import train
from agilerl.utils.utils import (
    calculate_vectorized_scores,
    initialPopulation,
    makeVectEnvs,
)
from PIL import Image, ImageDraw
from tqdm import trange

# Author: Michael Pratt
# License: MIT License
import gymnasium as gym


# %%
# Defining Hyperparameters
# ------------------------
# Before we commence training, it's easiest to define all of our hyperparameters in one dictionary. Below is an example of
# such for the Rainbow-DQN algorithm. For this example, we are training a single agent without hyperparameter optimisation,
# so we will not be performing mutations or tournament selection like we have in our other tutorials where we have. As this
# is the case, we do not need to define a mutatinos parameters dictionary.

# Initial hyperparameters
INIT_HP = {
    "POP_SIZE": 3,  # Population size
    "BATCH_SIZE": 256,  # Batch size
    "LR": 0.0000625,  # Learning rate
    "GAMMA": 0.999,  # Discount factor
    "MEMORY_SIZE": 200_000,  # Max memory buffer size
    "POLICY_FREQ": 2,  # Policy network update frequency
    "LEARN_STEP": 1,  # Learning frequency
    "N_STEP": 3,  # Step number to calculate td error
    "PER": True,  # Use prioritized experience replay buffer
    "ALPHA": 0.5,  # Prioritized replay buffer parameter
    "BETA": 0.7,  # Importance sampling coefficient
    "TAU": 0.005,  # For soft update of target parameters
    "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
    "NUM_ATOMS": 51,  # Unit number of support
    "V_MIN": 0.0,  # Minimum value of support
    "V_MAX": 200.0,  # Maximum value of support
    "NOISY": True,  # Add noise directly to the weights of the network
    # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    "CHANNELS_LAST": False,  # Use with RGB states
    "EPISODES": 1000,  # Number of episodes to train for
    "EVAL_EPS": 20,  # Number of episodes after which to evaluate the agent after
    "TARGET_SCORE": 200.0,  # Target score that will beat the environment
    "EVO_LOOP": 3,  # Number of evaluation episodes
    "MAX_STEPS": 500,  # Maximum number of steps an agent takes in an environment
    "TOURN_SIZE": 2,  # Tournament size
    "ELITISM": True,  # Elitism in tournament selection
    "EVO_EPOCHS": 20,
}

# Mutation parameters
MUT_P = {
    # Mutation probabilities
    "NO_MUT": 0.4,  # No mutation
    "ARCH_MUT": 0.2,  # Architecture mutation
    "NEW_LAYER": 0.2,  # New layer mutation
    "PARAMS_MUT": 0.2,  # Network parameters mutation
    "ACT_MUT": 0.2,  # Activation layer mutation
    "RL_HP_MUT": 0.2,  # Learning HP mutation
    "RL_HP_SELECTION": ["lr", "batch_size"],  # Learning HPs to choose from
    "MUT_SD": 0.1,  # Mutation strength
    "RAND_SEED": 42,  # Random seed
    # Define max and min limits for mutating RL hyperparams
    "MIN_LR": 0.0001,
    "MAX_LR": 0.01,
    "MIN_BATCH_SIZE": 8,
    "MAX_BATCH_SIZE": 1024,
}

# %%
# Create the Environment
# ----------------------
# In this particular tutorial, we will be focussing on the cart-pole environment as Rainbow-DQN can only be
# used with discrete action environments. The snippet below creates a vectorised environment and then assigns the
# correct values for ``state_dim`` and ``one_hot``, depending on whether the observation or action spaces are discrete
# or continuous.

env = makeVectEnvs("CartPole-v1", num_envs=8)  # Create environment
try:
    state_dim = env.single_observation_space.n  # Discrete observation space
    one_hot = True  # Requires one-hot encoding
except Exception:
    state_dim = env.single_observation_space.shape  # Continuous observation space
    one_hot = False  # Does not require one-hot encoding
try:
    action_dim = env.single_action_space.n  # Discrete action space
except Exception:
    action_dim = env.single_action_space.shape[0]  # Continuous action space

if INIT_HP[
    "CHANNELS_LAST"
]:  # Adjusts dimensions to be in accordance with PyTorch API (C, H, W), used with envs with RGB image states
    state_dim = (state_dim[2], state_dim[0], state_dim[1])

# %%
# Instantiate an Agent
# --------------------

# Set-up the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes
net_config = {"arch": "mlp", "h_size": [32, 32]}

# Define a Rainbow-DQN agent
# rainbow_dqn = RainbowDQN(
#     state_dim=state_dim,
#     action_dim=action_dim,
#     one_hot=one_hot,
#     net_config=net_config,
#     batch_size=INIT_HP["BATCH_SIZE"],
#     lr=INIT_HP["LR"],
#     learn_step=INIT_HP["LEARN_STEP"],
#     gamma=INIT_HP["GAMMA"],
#     tau=INIT_HP["TAU"],
#     beta=INIT_HP["BETA"],
#     n_step=INIT_HP["N_STEP"],
#     device=device,
# )
pop = initialPopulation(
    algo="Rainbow DQN",  # Algorithm
    state_dim=state_dim,  # State dimension
    action_dim=action_dim,  # Action dimension
    one_hot=one_hot,  # One-hot encoding
    net_config=net_config,  # Network configuration
    INIT_HP=INIT_HP,  # Initial hyperparameter
    population_size=INIT_HP["POP_SIZE"],  # Population size
    device=device,
)
print("Population instantiated", pop)
tournament = TournamentSelection(
    INIT_HP["TOURN_SIZE"],
    INIT_HP["ELITISM"],
    INIT_HP["POP_SIZE"],
    INIT_HP["EVO_EPOCHS"],
)

mutations = Mutations(
    algo="Rainbow DQN",
    no_mutation=MUT_P["NO_MUT"],
    architecture=MUT_P["ARCH_MUT"],
    new_layer_prob=MUT_P["NEW_LAYER"],
    parameters=MUT_P["PARAMS_MUT"],
    activation=MUT_P["ACT_MUT"],
    rl_hp=MUT_P["RL_HP_MUT"],
    rl_hp_selection=MUT_P["RL_HP_SELECTION"],
    mutation_sd=MUT_P["MUT_SD"],
    arch=net_config["arch"],
    rand_seed=MUT_P["RAND_SEED"],
    device=device,
)

# %%
# Experience Replay
# -----------------
# As mentioned in the summary of the algorithm , RainbowDQN makes use of multi-step learning and an additional
# prioritised replay buffer however, when training a population of agents, we recommend just using the standard
# AgileRL ``ReplayBuffer``. This is because agents within a population can share experiences from the standard
# replay buffer but not the n-step or prioritised buffer; the overall benefit of sharing experiences from the
# standard replay buffer outweighs the benefits of not sharing experiences and using an n-step buffer and a
# prioritised experience buffer.
#
# In this tutorial, we can make use of both the prioritised exerience replay and multi-step
# learning since we are only training a single agent and not making use of tournaments or mutations. Below is how
# you would define your memory and n_step_memory.

field_names = ["state", "action", "reward", "next_state", "termination"]
memory = PrioritizedReplayBuffer(
    action_dim,
    memory_size=INIT_HP["MEMORY_SIZE"],
    field_names=field_names,
    num_envs=8,
    alpha=INIT_HP["ALPHA"],
    gamma=INIT_HP["GAMMA"],
    device=device,
)
n_step_memory = MultiStepReplayBuffer(
    action_dim,
    memory_size=INIT_HP["MEMORY_SIZE"],
    field_names=field_names,
    num_envs=8,
    n_step=INIT_HP["N_STEP"],
    gamma=INIT_HP["GAMMA"],
    device=device,
)


# %%
# Training and Saving an Agent
# ----------------------------
#
# Using AgileRL ``train`` function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To train a single agent without performing tournament selection, mutations, and hyperparameter optimisation
# we can still use the AgileRL ``train`` function (Rainbow-DQN is an off-policy algorithm). We need to ensure
# that our single agent is passed to the function in a list (essentially a population of 1) and that we pass None
# for both the tournament and mutation arguments.

# Define parameters per and n_step
n_step = True if INIT_HP["N_STEP"] > 1 else False
per = INIT_HP["PER"]


trained_pop, pop_fitnesses = train(
    env=env,
    env_name="CartPole-v1",
    algo="RainbowDQN",
    pop=pop,
    memory=memory,
    n_step_memory=n_step_memory,
    INIT_HP=INIT_HP,
    MUT_P=MUT_P,
    swap_channels=INIT_HP["CHANNELS_LAST"],
    n_episodes=INIT_HP["EPISODES"],
    evo_loop=INIT_HP["EVO_LOOP"],
    target=INIT_HP["TARGET_SCORE"],
    evo_epochs=INIT_HP["EVAL_EPS"],
    n_step=n_step,
    per=per,
    noisy=INIT_HP["NOISY"],
    tournament=tournament,
    mutation=mutations,
    wb=False,  # Boolean flag to record run with Weights & Biases
    checkpoint=INIT_HP["EPISODES"],
    checkpoint_path="RainbowDQN.pt",
)

# %%
# Using a custom training loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If we wanted to have more control over the training process, it is also possible to write our own custom
# training loops to train our agents. The training loop below can be used alternatively to the above ``train_on_policy``
# function and is an example of how we might choose to train an AgileRL agent.

# total_steps = 0

# n_step = True if INIT_HP["N_STEP"] > 1 else False
# per = INIT_HP["PER"]

# for episode in trange(INIT_HP["EPISODES"]):
#     state = env.reset()[0]  # Reset environment at start of episode
#     rewards, terminations, truncs = [], [], []
#     score = 0
#     for step in range(INIT_HP["MAX_STEPS"]):
#         if INIT_HP["CHANNELS_LAST"]:
#             state = np.moveaxis(state, [-1], [-3])
#         # Get next action from agent
#         action = rainbow_dqn.getAction(state)
#         next_state, reward, done, trunc, _ = env.step(action)  # Act in environment

#         if INIT_HP["CHANNELS_LAST"]:
#             one_step_transition = n_step_memory.save2memoryVectEnvs(
#                 state,
#                 action,
#                 reward,
#                 np.moveaxis(next_state, [-1], [-3]),
#                 done,
#             )
#         else:
#             one_step_transition = n_step_memory.save2memoryVectEnvs(
#                 state,
#                 action,
#                 reward,
#                 next_state,
#                 done,
#             )
#         if one_step_transition:
#             memory.save2memoryVectEnvs(*one_step_transition)

#         # Learn according to learning frequency
#         if per:
#             fraction = min((step + 1) / INIT_HP["MAX_STEPS"], 1.0)
#             rainbow_dqn.beta += fraction * (1.0 - rainbow_dqn.beta)

#         # Learn according to learning frequency
#         if (
#             memory.counter % rainbow_dqn.learn_step == 0
#             and len(memory) >= rainbow_dqn.batch_size
#         ):
#             # Sample replay buffer
#             # Learn according to agent's RL algorithm

#             experiences = memory.sample(rainbow_dqn.batch_size, rainbow_dqn.beta)
#             if n_step_memory is not None:
#                 n_step_experiences = n_step_memory.sample_from_indices(experiences[6])
#                 experiences += n_step_experiences
#             idxs, priorities = rainbow_dqn.learn(experiences, n_step=n_step, per=per)
#             memory.update_priorities(idxs, priorities)

#         terminations.append(done)
#         rewards.append(reward)
#         truncs.append(trunc)
#         state = next_state

#         rainbow_dqn.scores.append(score)

#         rainbow_dqn.steps[-1] += step
#         total_steps += step

#     scores = calculate_vectorized_scores(np.array(rewards), np.array(terminations))
#     score = np.mean(scores)

#     rainbow_dqn.scores.append(score)

#     rainbow_dqn.steps[-1] += INIT_HP["MAX_STEPS"]
#     total_steps += INIT_HP["MAX_STEPS"]

#     # Now evolve population if necessary
#     if (episode + 1) % INIT_HP["EVAL_EPS"] == 0:
#         # Evaluate population
#         fitness = rainbow_dqn.test(
#             env,
#             swap_channels=INIT_HP["CHANNELS_LAST"],
#             max_steps=INIT_HP["MAX_STEPS"],
#             loop=INIT_HP["EVO_LOOP"],
#         )

#         fitness = "%.2f" % fitness
#         avg_fitness = "%.2f" % np.mean(rainbow_dqn.fitness[-100:])
#         avg_score = "%.2f" % np.mean(rainbow_dqn.scores[-100:])
#         num_steps = rainbow_dqn.steps[-1]

#         print(
#             f"""
#             --- Epoch {episode + 1} ---
#             Fitness:\t\t{fitness}
#             100 fitness avgs:\t{avg_fitness}
#             100 score avgs:\t{avg_score}
#             Steps:\t\t{num_steps}
#             """,
#             end="\r",
#         )

#     if episode + 1 == INIT_HP["EPISODES"]:
#         # Save the trained algorithm at the end of the training loop
#         save_path = f"RainbowDQN_0_{INIT_HP['EPISODES']}.pt"
#         rainbow_dqn.saveCheckpoint(save_path)


# %%
# Loading an Agent for Inference and Rendering your Solved Environment
# --------------------------------------------------------------------
# Once we have trained and saved an agent, we may want to then use our trained agent for inference. Below outlines
# how we would load a saved agent and how it can then be used in a testing loop.

# %%
# Load agent
# ~~~~~~~~~~

# Instantiate a RainbowDQN object
# rainbow_dqn = RainbowDQN(
#     state_dim=state_dim,
#     action_dim=action_dim,
#     one_hot=one_hot,
# )

# # Load in the saved model
# rainbow_dqn.loadCheckpoint(save_path)

# %%
# Define function to label image with episode number
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The below function adds the episode number to associated image frames and will allow us to see how the agent performed in
# each episode.


def label_frame(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"Episode: {episode_num+1}", fill=text_color
    )

    return im


# %%
# Test loop for inference
# ~~~~~~~~~~~~~~~~~~~~~~~
# rewards = []
# frames = []
# testing_eps = 5
# test_env = gym.make("CartPole-v1", render_mode="rgb_array")
# with torch.no_grad():
#     for ep in range(testing_eps):
#         state = test_env.reset()[0]  # Reset environment at start of episode
#         score = 0

#         for step in range(INIT_HP["MAX_STEPS"]):
#             # If your state is an RGB image
#             if INIT_HP["CHANNELS_LAST"]:
#                 state = np.moveaxis(state, [-1], [-3])

#             # Get next action from agent
#             action, *_ = rainbow_dqn.getAction(state)

#             # Save the frame for this step and append to frames list
#             frame = test_env.render()
#             frames.append(label_frame(frame, episode_num=ep))

#             # Take the action in the environment
#             state, reward, terminated, truncated, _ = test_env.step(
#                 action
#             )  # Act in environment

#             # Collect the score of environment 0
#             score += reward

#             # Break if environment 0 is done or truncated
#             if terminated or truncated:
#                 break

#         # Collect and print episodic reward
#         rewards.append(score)
#         print("-" * 15, f"Episode: {ep}", "-" * 15)
#         print("Episodic Reward: ", rewards[-1])

#     test_env.close()

# %%
# Save test episosdes as a gif
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gif_path = "./videos/"
# os.makedirs(gif_path, exist_ok=True)
# imageio.mimwrite(
#     os.path.join("./videos/", "rainbow_dqn_cartpole.gif"), frames, duration=10
# )
# mean_fitness = np.mean(rewards)
