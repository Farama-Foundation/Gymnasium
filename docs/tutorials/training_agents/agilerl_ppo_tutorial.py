"""
AgileRL PPO Implementation
==========================

"""

# %%
# In this tutorial, we will be training and then deploying an AgileRL PPO agent to beat the 
# Gymnasium continuous lunar lander environment. AgileRL is a deep reinforcement learning 
# library, focussed on improving the RL training process through evolutionary hyperparameter 
# optimisation (HPO), which has resulted in 10x faster HPO compared to other popular deep RL
# libraries. Check out the AgileRL repository for more information about the library.

# %%
# Dependencies
# ------------
#

# %%
# Let's first import a few dependencies we'll need.
#

# Author: Michael Pratt
# License: MIT License
import torch
from tqdm import trange
import os

from agilerl.utils.utils import initialPopulation
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import initialPopulation, makeVectEnvs, printHyperparams
from agilerl.algorithms.ppo import PPO

# %%
# Defining Hyperparameters
# ------------------------
# Explain that agilerl needs inital hps defined in a dict at the start

# Initial hyperparameters
INIT_HP = {
        "POPULATION_SIZE": 6,  # Population size
        "DISCRETE_ACTIONS": False,  # Discrete action space
        "BATCH_SIZE": 128,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
        "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
        "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
        "ENT_COEF": 0.01,  # Entropy coefficient
        "VF_COEF": 0.5,  # Value function coefficient
        "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
        "TARGET_KL": None,  # Target KL divergence threshold
        "UPDATE_EPOCHS": 4,  # Number of policy update epochs
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "EPISODES": 1000, # Number of episodes to train for
        "EVO_EPOCHS": 20, # Evolution frequency, i.e. evolve after every 20 episodes
        "TARGET_SCORE": 200., # Target score that will beat the environment
        "EVO_LOOP": 3, # Number of evalutation episodes
        "MAX_STEPS": 500, # Maximum number of steps an agent takes in an environment
    }

# Mutation parameters
MUT_P = {
        "NO_MUT": 0.4,                            # No mutation
        "ARCH_MUT": 0.2,                          # Architecture mutation
        "NEW_LAYER": 0.2,                         # New layer mutation
        "PARAMS_MUT": 0.2,                        # Network parameters mutation
        "ACT_MUT": 0.2,                           # Activation layer mutation
        "RL_HP_MUT": 0.2,                         # Learning HP mutation
        "RL_HP_SELECTION": [lr, batch_size],      # Learning HPs to choose from
        "MUT_SD": 0.1,                            # Mutation strength
        "RAND_SEED": 42,                          # Random seed
    # Define max and min limits for mutating RL hyperparams
        "MIN_LR": 0.0001,                         
        "MAX_LR": 0.01,
        "MIN_BATCH_SIZE": 8,
        "MAX_BATCH_SIZE": 1024,
}
# %%
# Create the Environment
# ----------------------
# Bit of chat about continuous environment and how PPO can deal with discrete and continuous

env = makeVectEnvs("LunarLanderContinuous-v2", num_envs=8)  # Create environment
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

if INIT_HP["CHANNELS_LAST"]: # Adjusts dimensions to be in accordance with PyTorch API (C, H, W), used with envs with RGB image states
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

# %%
# Create a Population of Agents
# -----------------------------
# Explain here why agilerl needs a population of agents - go into a bit more detail about evolutionary 
# hpo works.

# Set-up the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes
net_config = {"arch": "mlp", "h_size": [64, 64]}

# Define a population
pop = initialPopulation(
        algo="PPO",  # Algorithm
        state_dim=state_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=one_hot,  # One-hot encoding
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        device=device,
    )

# %% 
# Creating Mutations and Tournament objects
# -----------------------------------------
# Nice segway from previous cell - create mut and tourn objects this will enable the populations to
# mutate, evolve, and then selection of the elit for the next generation of mutations
tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVO_EPOCHS"],
    )

mutations = Mutations(
    algo=INIT_HP["ALGO"],
    no_mutation=MUTATION_PARAMS["NO_MUT"],
    architecture=MUTATION_PARAMS["ARCH_MUT"],
    new_layer_prob=MUTATION_PARAMS["NEW_LAYER"],
    parameters=MUTATION_PARAMS["PARAMS_MUT"],
    activation=MUTATION_PARAMS["ACT_MUT"],
    rl_hp=MUTATION_PARAMS["RL_HP_MUT"],
    rl_hp_selection=MUTATION_PARAMS["RL_HP_SELECTION"],
    mutation_sd=MUTATION_PARAMS["MUT_SD"],
    arch=NET_CONFIG["arch"],
    rand_seed=MUTATION_PARAMS["RAND_SEED"],
    device=device,
)

# %% 
# Training and Saving an Agent
# ----------------------------
#
# Using AgileRL ``train`` function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The simplest way to train an AgileRL agent is to use one of the implemented AgileRL train functions.
# Given that PPO is an on-policy algorithm, we can make use of the ``train_on_policy`` function. This 
# training function will orchestrate the training and hyperparameter optimisation process, removing the 
# the need to implement a training loop and will return a trained population, as well as the associated 
# fitnesses (fitness is each agents test scores on the environment).

trained_pop, pop_fitnesses = train_on_policy(
    env=env,
    env_name="LunarLanderContinuous-v2",
    algo="PPO",
    pop=pop,
    INIT_HP=INIT_HP,
    MUT_P=MUT_P,
    swap_channels=INIT_HP["CHANNELS_LAST"],
    n_episodes=INIT_HP["EPISODES"],
    evo_epochs=INIT_HP["EVO_EPOCHS"],
    evo_loop=INIT_HP["EVO_LOOP"]
    target=INIT_HP["TARGET_SCORE"],
    tournament=tournament,
    mutation=mutations,
    wb=False, # Boolean flag to record run with Weights & Biases
)

# Using a custom training loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If you would like to have more control over the training process, it is also possible to write your own custom 
# training loops to train your agents. The training loop below is to be used alternatively to the above ``train_on_policy``
# function and is an example of how you might choose to train an AgileRL agent.

    for episode in trange(INIT_HP["EPISODES"]):
        for agent in pop:  # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            score = 0

            states = []
            actions = []
            log_probs = []
            rewards = []
            dones = []
            values = []

            for step in range(max_steps):
                if INIT_HP["CHANNELS_LAST"]:
                    state = np.moveaxis(state, [-1], [-3])

                # Get next action from agent
                action, log_prob, _, value = agent.getAction(state)
                next_state, reward, done, trunc, _ = env.step(
                    action
                )  # Act in environment

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value)

                state = next_state
                score += reward

            if INIT_HP["CHANNELS_LAST"]:
                next_state = np.moveaxis(next_state, [-1], [-3])

            agent.scores.append(score)

            experiences = (
                states,
                actions,
                log_probs,
                rewards,
                dones,
                values,
                next_state,
            )
            # Learn according to agent's RL algorithm
            agent.learn(experiences)

            agent.steps[-1] += step + 1

        # Now evolve population if necessary
        if (episode + 1) % evo_epochs == 0:
            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
                    swap_channels=INIT_HP["CHANNELS_LAST"],
                    max_steps=max_steps,
                    loop=evo_loop,
                )
                for agent in pop
            ]

            print(f"Episode {episode+1}/{max_episodes}")
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(
                f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}'
            )

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            if episode + 1 != INIT_HP["EPISODES"]:
                pop = mutations.mutation(pop)

# Save the trained algorithm
path = "./models/PPO_elite"
filename = "PPO_trained_agent.pt"
os.makedirs(path, exist_ok=True)
save_path = os.path.join(path, filename)
elite.saveCheckpoint(save_path)


# %% 
# Loading an Agent for Inference
# ------------------------------
# Once you have trained and saved an agent, you may want to then use your trained agent for inference. Below outlines
# how you would load a saved agent and how it can then be incorporated into a testing loop.

# Instantiate a PPO object
ppo = PPO(state_dim=state_dim,
          action_dim=action_dim,
          one_hot=one_hot,
          discrete_actions=INIT_HP["DISCRETE_ACTIONS"])

# Load in the saved model
ppo.loadCheckpoint(path)

# Test loop for inference
rewards = []
testing_eps = 5
with torch.no_grad():

    for _ in range(testing_eps):
        state = env.reset()[0]  # Reset environment at start of episode
        score = 0

        for step in range(INIT_HP["MAX_STEPS"]):
            if INIT_HP["CHANNELS_LAST"]:
                state = np.moveaxis(state, [-1], [-3])

            # Get next action from agent
            action, *_ = agent.getAction(state)
            state, reward, done, trunc, _ = env.step(
                action
            )  # Act in environment

            score += reward

            if done[0] or trunc[0]:
                break

    rewards.append(score)
    
mean_fitness = np.mean(rewards)


            













