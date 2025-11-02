---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/img/gymnasium-text.png
:alt: Gymnasium Logo
```

```{project-heading}
An API standard for reinforcement learning with a diverse collection of reference environments
```

```{figure} _static/videos/box2d/lunar_lander.gif
   :alt: Lunar Lander
   :width: 500
```

**Gymnasium is a maintained fork of OpenAI’s Gym library.** The Gymnasium interface is simple, pythonic, and capable of representing general RL problems, and has a [migration guide](introduction/migration_guide) for old Gym environments:

```{code-block} python
import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()

import numpy as np

rows, cols = 5, 5
gamma = 0.9

A, A_prime = (0, 1), (4, 1)
B, B_prime = (0, 3), (2, 3)

actions = {
    0: (-1, 0),  # north
    1: (1, 0),   # south
    2: (0, -1),  # west
    3: (0, 1)    # east
}

def next_state_and_reward(state, action, reward_config):
    i, j = state
    if state == A: return A_prime, reward_config["A"]
    if state == B: return B_prime, reward_config["B"]
    di, dj = actions[action]
    ni, nj = i + di, j + dj
    if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
        return state, reward_config["off"]
    return (ni, nj), reward_config["normal"]

def random_policy(state):
    return np.random.choice([0, 1, 2, 3])

# Monte Carlo value estimation
V = np.zeros((rows, cols))
returns_count = np.zeros((rows, cols))

for episode in range(500):
    initial_state = (np.random.randint(rows), np.random.randint(cols))
    history = []
    state = initial_state
    for _ in range(1000):
        action = random_policy(state)
        next_s, reward = next_state_and_reward(state, action, reward_config_1)
        history.append((state, action, reward, next_s))
        state = next_s
    
    # Backward pass for returns
    G = 0
    for t in reversed(range(len(history))):
        state, action, reward, next_state = history[t]
        G = reward + gamma * G  # THE KEY LINE!
        i, j = state
        V[i, j] += G
        returns_count[i, j] += 1

V = np.divide(V, returns_count, out=np.zeros_like(V), where=returns_count != 0)
```

```{toctree}
:hidden:
:caption: Introduction

introduction/basic_usage
introduction/train_agent
introduction/create_custom_env
introduction/record_agent
introduction/speed_up_env
introduction/migration_guide
```

```{toctree}
:hidden:
:caption: API

api/env
api/registry
api/spaces
api/wrappers
api/vector
api/utils
api/functional
```

```{toctree}
:hidden:
:caption: Environments

environments/classic_control
environments/box2d
environments/toy_text
environments/mujoco
environments/atari
environments/third_party_environments
```

```{toctree}
:hidden:
:glob:
:caption: Tutorials

tutorials/**/index
tutorials/third-party-tutorials
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/Gymnasium>
Paper <https://arxiv.org/abs/2407.17032>
gymnasium_release_notes/index
gym_release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/Gymnasium/blob/main/docs/README.md>
```

