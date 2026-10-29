---
layout: "contents"
title: Speeding Up Training
---

# Speeding Up Training

Reinforcement learning is a computationally difficult problem that is both sample inefficient and difficult to scale to more complex environments.
In this page, we are going to talk about general strategies for speeding up training: vectorizing environments, optimizing the training loop and algorithmic heuristics.

## Vectorized environments

```{eval-rst}
.. py:currentmodule:: gymnasium

Stepping a single environment limits the number of samples per second an agent can collect to the speed of that environment. A vectorized environment runs several copies of the same environment and steps them with a single call, which increases the number of samples per second and hands the training algorithm a batch of transitions. Gymnasium provides two classes for this, :class:`gymnasium.vector.SyncVectorEnv` and :class:`gymnasium.vector.AsyncVectorEnv`, as well as :meth:`make_vec` to create either of them from an environment id.
```

```python
import gymnasium as gym

# Four CartPole environments stepped together in the current process
envs = gym.make_vec("CartPole-v1", num_envs=4, vectorization_mode="sync")

observations, info = envs.reset(seed=42)
print(observations.shape)  # (4, 4): one observation per environment
print(envs.action_space)  # MultiDiscrete([2 2 2 2])

actions = envs.action_space.sample()
print(actions.shape)  # (4,): one action per environment

observations, rewards, terminated, truncated, info = envs.step(actions)
print(rewards.shape)  # (4,): one reward per environment

envs.close()
```

```{eval-rst}
.. py:currentmodule:: gymnasium

The observations, rewards and the ``terminated`` / ``truncated`` flags carry one entry per sub-environment, and every value in ``info`` is an array with one entry per sub-environment.

The vector environment also resets sub-environments by itself. Whenever a sub-environment returns ``terminated`` or ``truncated``, the vector environment resets it before the next ``step`` call, so the reset logic of a single-environment training loop is usually removed when the loop is moved onto a vector environment. Because of this, a batch can mix continuing sub-environments with one or more that were just reset, and the ``terminated`` and ``truncated`` arrays are what the training algorithm should use to mask the data of each sub-environment. See the `vector environments API <https://farama.org/Vector-Autoreset-Mode>`_ for the details of the autoreset modes.
```

### Synchronous or asynchronous

| Class | Execution | Overhead | When to use it |
| --- | --- | --- | --- |
| `SyncVectorEnv` | Sub-environments step one after another in the current process | None | Fast Python environments, debugging, and algorithms that only need the batched data |
| `AsyncVectorEnv` | Every sub-environment runs in its own process | Inter-process communication for observations and actions | Environments whose own code (simulation, rendering, Python loops) is the bottleneck |

```python
import gymnasium as gym

if __name__ == "__main__":
    # Eight environments in eight processes. This is usually only faster
    # when the environment is the bottleneck, not the training loop.
    envs = gym.make_vec(
        "CartPole-v1", num_envs=8, vectorization_mode="async"
    )
    observations, info = envs.reset(seed=42)
    print(observations.shape)  # (8, 4)
    envs.close()
```

```{eval-rst}
.. warning::

   :class:`gymnasium.vector.AsyncVectorEnv` starts one process per sub-environment. Platforms that create processes with ``spawn`` (Windows and macOS) require the entry point of the script to be guarded with ``if __name__ == "__main__":``, as in the example above, otherwise each process re-imports and re-runs the script.
```

```{eval-rst}
.. py:currentmodule:: gymnasium

The number of sub-environments is usually chosen based on the number of available CPU cores. Throughput stops improving once the CPU is saturated, and very large numbers of environments can also make training less stable because every update is computed from more independent episodes. Synchronous vector environments avoid the communication cost entirely, and they are often the better default for small or fast environments.
```

## Optimizing training

```{eval-rst}
.. py:currentmodule:: gymnasium

After the environments, the training loop is usually the next bottleneck. Rendering is expensive, so create the environments with ``render_mode=None`` and only wrap them with the video recording wrappers when a recording is actually needed, rather than for every training run.

For deep reinforcement learning, most of the remaining time is spent in the neural network and in moving data between the CPU and the accelerator. Compiling the model or the update step reduces the Python overhead of the step, and keeping the observations, actions and targets on the accelerator avoids a round trip to the CPU for every batch.
```

<!-- test: skip -->
```python
import torch

# `model` is a torch.nn.Module, e.g. the policy network.
# The first call is slow, the following calls are faster.
model = torch.compile(model)
```

```{eval-rst}
.. py:currentmodule:: gymnasium

The equivalent for JAX is to decorate the update step with ``jax.jit`` and to keep the parameters and the batch on the accelerator. Compilation only pays off when the shapes of the inputs do not change between calls, which is another reason to step a fixed batch of environments.
```

<!-- test: skip -->
```python
import jax

# JIT the update step and keep the batch on the accelerator
update_step = jax.jit(update_step)
```

## Algorithmic heuristics

Academic researchers are consistently exploring new optimizations that reduce the number of samples required to train an agent.
Sample-efficient reinforcement learning is a specialist sub-field of reinforcement learning that studies optimizations for training algorithms and environment heuristics that reduce the number of agent observations needed to maximize performance.

Common heuristics that can reduce the number of environment steps are:

- **Reward shaping**: adding intermediate rewards for progress towards the task, so the agent does not have to discover the goal by chance.
- **Frame skipping and action repetition**: repeating an action over several environment steps, which reduces the number of decisions the agent has to make.
- **Observation normalization**: rescaling observations with wrappers such as [NormalizeObservation](/api/wrappers/table), so that a single learning rate works across features.
- **Curriculum learning**: starting with an easier version of the task and increasing the difficulty as the agent improves.

As the field is consistently improving, we refer readers to survey papers and the latest research for the algorithmic improvements that are currently the most effective.

## Where to go next

- [Vector environments API](/api/vector/) for the full vector environment API.
- [Vector A2C tutorial](/tutorials/training_agents/vector_a2c/) for a training loop that uses vectorized environments.
- [Train an Agent](train_agent) for a first end-to-end training example.
- [Record an Agent](record_agent) for recording videos and episode statistics.
