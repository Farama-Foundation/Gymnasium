# Tutorials

<div class="sphx-glr-thumbnails"></div>

# Training Agents

The most common application of Gymnasium is for training RL agents. Therefore, these tutorials aim to show a range of example implementations for different environments.

Each tutorial trains a small agent from scratch with a compact, readable algorithm, so the full loop
(collecting experience, updating the policy or value function, and evaluating the result) stays
visible instead of being hidden behind a training library.

## What the tutorials cover

- **Solving Blackjack with Tabular Q-Learning** - Q-learning with an epsilon-greedy policy on
  `Blackjack-v1`, including a heatmap of the learned state-action values.
- **Solving FrozenLake with Tabular Q-Learning** - the same algorithm on `FrozenLake-v1`, compared
  across map sizes.
- **Action Masking in the Taxi Environment** - tabular Q-learning on `Taxi-v4` with and without the
  `action_mask` that the environment returns in `info`.
- **Training using REINFORCE for Mujoco** - a PyTorch implementation of REINFORCE, a Monte-Carlo
  policy gradient method, on `InvertedPendulum-v5`.
- **Speeding up A2C Training with Vector Envs** - an A2C implementation trained with
  `SyncVectorEnv` and `AsyncVectorEnv` on `LunarLander-v3`, showing how running environments in
  parallel increases throughput.

## Requirements

The tutorials require Python 3.11 or newer, `gymnasium` with the `toy-text` extra (for Blackjack,
FrozenLake and Taxi), and the plotting and progress-bar packages used by the training scripts:

```console
pip install "gymnasium[toy-text]" matplotlib pandas seaborn tqdm
```

The two deep reinforcement learning tutorials also need PyTorch and the extra for the environment
they train on:

```console
pip install "gymnasium[mujoco,box2d]" torch
```

*Training using REINFORCE for Mujoco* trains on `InvertedPendulum-v5` from the `mujoco` extra and
*Speeding up A2C Training with Vector Envs* trains on `LunarLander-v3` from the `box2d` extra.

## Running a tutorial locally

Every tutorial page ends with a *Download Python source code* link. Save the file and run it from a
terminal:

```console
python blackjack_q_learning.py
```

Training takes time: Blackjack trains for 100,000 episodes, FrozenLake for 2,000 episodes per run
across several map sizes, Taxi for 5,000 episodes in each of 12 runs, REINFORCE for 5,000 episodes and A2C for 1,000 updates in several parallel
environments. Lower the episode, update and timestep counts while experimenting, and raise them again
for a final run.

The vector environment tutorial explains how to wrap the training script in
`if __name__ == "__main__":`, which `AsyncVectorEnv` needs on platforms that start processes with
`spawn` (macOS and Windows).

## Where to go next

- [Gymnasium Basics](https://gymnasium.farama.org/tutorials/gymnasium_basics/) covers the
  `reset` / `step` loop and wrappers if you are new to Gymnasium.
- [Vector environments](https://gymnasium.farama.org/api/vector/) documents the vector API used by
  the A2C tutorial.

<div class="sphx-glr-thumbnails"><div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates how to use action masking in the Taxi environment to improve reinfor...">  <div class="sphx-glr-thumbnail-title">Action Masking in the Taxi Environment</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This tutorial trains an agent for BlackJack using tabular Q-learning. /introduction/train_agent...">  <div class="sphx-glr-thumbnail-title">Solving Blackjack with Tabular Q-Learning</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This tutorial trains an agent for FrozenLake using tabular Q-learning.">  <div class="sphx-glr-thumbnail-title">Solving Frozenlake with Tabular Q-Learning</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This tutorial implements REINFORCE with neural networks for a MuJoCo environment.">  <div class="sphx-glr-thumbnail-title">Training using REINFORCE for Mujoco</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates training with vector environments to speed it up. It assumes familia...">  <div class="sphx-glr-thumbnail-title">Speeding up A2C Training with Vector Envs</div>
</div></div>

# Gymnasium Basics

The aim of these tutorials is to showcase the fundamental API of Gymnasium to help users implement it.

Each tutorial is a self-contained Python script: its page shows the code next to the output, and the
script can be downloaded from the same page to run and modify locally.

## What the tutorials cover

- **Make your own custom environment** - generate a new environment from the Farama
  `gymnasium-env-template` with Copier, then implement its spaces, transitions, rendering,
  registration and tests.
- **Implementing custom wrappers** - subclass `ObservationWrapper`, `ActionWrapper`,
  `RewardWrapper` and `Wrapper` to change the observations, actions or rewards of an existing
  environment, and where to find the function-based and vector wrappers.
- **Handling time limits** - the difference between `terminated` and `truncated`, why
  bootstrapping should stop only on `terminated`, and how this works with vector environments.
- **Load custom quadruped robot environments** - pass an MJCF or URDF model to a MuJoCo environment
  with the `xml_file` argument, then tune its simulation, termination, reward and observation
  parameters.

## Requirements

The tutorials require Python 3.11 or newer with `gymnasium` installed:

```console
pip install gymnasium
```

Three of the tutorials also use optional dependencies:

- `pip install copier "gymnasium[classic-control]"` for *Make your own custom environment*, which
  walks through the Copier template and renders the example environment with `pygame`.
- `pip install "gymnasium[box2d,mujoco]"` for *Implementing custom wrappers*, which wraps
  `LunarLanderContinuous-v3` and `Reacher-v5`.
- `pip install "gymnasium[mujoco]"` for *Load custom quadruped robot environments*, which wraps
  `Ant-v5`.

## Running a tutorial locally

Every tutorial page ends with a *Download Python source code* link. Save the file and run it from a
terminal, for example:

```console
python implementing_custom_wrappers.py
```

The tutorials are written to be read as much as run: the printed output and figures are part of the
explanation, so the rendered page is the primary reference and the script is there to experiment
with.

## Where to go next

- [Basic usage](https://gymnasium.farama.org/introduction/basic_usage/) introduces the
  `reset` / `step` loop, spaces and wrappers that every tutorial builds on.
- The [API reference](https://gymnasium.farama.org/api/env/) documents the environment API in
  full.

<div class="sphx-glr-thumbnails"><div class="sphx-glr-thumbcontainer" tooltip="This tutorial shows how to create new environment and links to relevant useful wrappers, utilit...">  <div class="sphx-glr-thumbnail-title">Make your own custom environment</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This tutorial explains how time limits should be correctly handled with termination and truncat...">  <div class="sphx-glr-thumbnail-title">Handling Time Limits</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial we will describe how to implement your own custom wrappers.">  <div class="sphx-glr-thumbnail-title">Implementing Custom Wrappers</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial we create a MuJoCo quadruped walking environment using a model file (ending in...">  <div class="sphx-glr-thumbnail-title">Load custom quadruped robot environments</div>
</div></div>
