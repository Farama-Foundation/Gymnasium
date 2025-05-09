---
layout: "contents"
title: Basic Usage
firstpage:
---

# Speeding Up Training

Reinforcement Learning can be a computationally difficult problem that is both sample inefficient and difficult to scale to more complex environments.
In this page, we are going to talk about general strategies for speeding up training: vectorizing environments, optimizing training and algorithmic heuristics.

## Vectorized environments

```{eval-rst}
.. py:currentmodule:: gymnasium

Normally in training, agents will sample from a single environment limiting the number of steps (samples) per second to the speed of the environment. Training can be substantially increased through acting in multiple environments at the same time, referred to as vectorized environments where multiple instances of the same environment run in parallel (on multiple CPUs). Gymnasium provide two built in classes to vectorize most generic environments: :class:`gymnasium.vector.SyncVectorEnv` and :class:`gymnasium.vector.AsyncVectorEnv` which can be easily created with :meth:`gymnasium.make_vec`.

It should be noted that vectorizing environments might require changes to your training algorithm and can cause instability in training for very large numbers of sub-environments.
```

## Optimizing training

Speeding up training can generally be achieved through optimizing your code, in particular, for deep reinforcement learning that use GPUs in training through the need to transfer data to and from RAM and the GPU memory.

For code written in PyTorch and Jax, they provide the ability to `jit` (just in time compilation) the code order for CPU, GPU and TPU (for jax) to decrease the training time taken.

## Algorithmic heuristics

Academic researchers are consistently exploring new optimizations to improve agent performance and reduce the number of samples required to train an agent.
In particular, sample efficient reinforcement learning is a specialist sub-field of reinforcement learning that explores optimizations for training algorithms and environment heuristics that reduce the number of agent observation need for an agent to maximise performance.
As the field is consistently improving, we refer readers to find to survey papers and the latest research to know what the most efficient algorithmic improves that exist currently.
