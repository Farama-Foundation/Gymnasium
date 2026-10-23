Training Agents
===============

.. _gallery_section_name:

The most common application of Gymnasium is for training RL agents. Therefore, these tutorials aim to show a range of example implementations for different environments.

Each tutorial trains a small agent from scratch with a compact, readable algorithm, so the full loop
(collecting experience, updating the policy or value function, and evaluating the result) stays
visible instead of being hidden behind a training library.

What the tutorials cover
------------------------

- **Solving Blackjack with Tabular Q-Learning** - Q-learning with an epsilon-greedy policy on
  ``Blackjack-v1``, including a heatmap of the learned state-action values.
- **Solving FrozenLake with Tabular Q-Learning** - the same algorithm on ``FrozenLake-v1``, compared
  across map sizes and exploration schedules.
- **Solving Lunar Lander with Deep Q-Learning (DQN)** - a PyTorch implementation of DQN with
  experience replay and a target network on ``LunarLander-v3``, including how ``terminated`` and
  ``truncated`` enter the Q-learning target.
- **Training using REINFORCE for Mujoco** - a PyTorch implementation of REINFORCE, a Monte-Carlo
  policy gradient method, on ``InvertedPendulum-v4``.
- **Speeding up A2C Training with Vector Envs** - an A2C implementation trained with
  ``SyncVectorEnv`` and ``AsyncVectorEnv`` on ``LunarLander-v3``, showing how running environments in
  parallel increases throughput.

Requirements
------------

The tutorials require Python 3.11 or newer, ``gymnasium``, and the plotting and progress-bar packages
used by the training scripts:

.. code:: console

    pip install gymnasium matplotlib pandas seaborn tqdm

The deep reinforcement learning tutorials also need PyTorch and the extra for the environment
they train on:

.. code:: console

    pip install "gymnasium[mujoco,box2d]" torch

*Training using REINFORCE for Mujoco* trains on ``InvertedPendulum-v4`` from the ``mujoco`` extra, while
*Solving Lunar Lander with Deep Q-Learning (DQN)* and *Speeding up A2C Training with Vector Envs* train on
``LunarLander-v3`` from the ``box2d`` extra.

Running a tutorial locally
--------------------------

Every tutorial page ends with a *Download Python source code* link. Save the file and run it from a
terminal:

.. code:: console

    python blackjack_q_learning.py

Training takes time: Blackjack trains for 100,000 episodes, FrozenLake for 2,000 episodes per run
across several map sizes, DQN for up to 2,000 episodes (it stops early once solved), REINFORCE for
5,000 episodes and A2C for 1,000 updates in several parallel environments. Lower the episode, update
and timestep counts while experimenting, and raise them again for a final run.

The vector environment tutorial explains how to wrap the training script in
``if __name__ == "__main__":``, which ``AsyncVectorEnv`` needs on platforms that start processes with
``spawn`` (macOS and Windows).

Where to go next
----------------

- `Gymnasium Basics <https://gymnasium.farama.org/tutorials/gymnasium_basics/>`_ covers the
  ``reset`` / ``step`` loop and wrappers if you are new to Gymnasium.
- `Vector environments <https://gymnasium.farama.org/api/vector/>`_ documents the vector API used by
  the A2C tutorial.
