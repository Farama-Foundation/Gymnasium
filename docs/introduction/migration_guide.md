---
layout: "contents"
title: Migration Guide
---

# Gym Migration Guide

## Who Should Read This Guide?

**If you're new to Gymnasium**: You can probably skip this page! This guide is for users migrating from older versions of OpenAI Gym. If you're just starting with RL, head to [Basic Usage](basic_usage) instead.

**If you're migrating from OpenAI Gym**: This guide will help you update your code to work with Gymnasium. The changes are significant but straightforward once you understand the reasoning behind them.

**If you're updating old tutorials**: Many online RL tutorials use the old v0.21 API. This guide shows you how to modernize that code.

## Why Did the API Change?

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

Gymnasium is a fork of `OpenAI Gym v0.26 <https://github.com/openai/gym/releases/tag/0.26.2>`_, which introduced breaking changes from `Gym v0.21 <https://github.com/openai/gym/releases/tag/v0.21.0>`_. These changes weren't made lightly - they solved important problems that made RL research and development more difficult.

The main issues with the old API were:

- **Ambiguous episode endings**: The single ``done`` flag couldn't distinguish between "task completed" and "time limit reached"
- **Inconsistent seeding**: Random number generation was unreliable and hard to reproduce
- **Rendering complexity**: Switching between visual modes was unnecessarily complicated
- **Reproducibility problems**: Subtle bugs made it difficult to reproduce research results

For environments that can't be updated, see the compatibility guide section below.
```

## Quick Reference: Complete Changes Table

| **Component**            | **v0.21 (Old)**                                   | **v0.26+ (New)**                                              | **Impact**      |
|--------------------------|---------------------------------------------------|---------------------------------------------------------------|-----------------|
| **Package Import**       | `import gym`                                      | `import gymnasium as gym`                                     | All code        |
| **Environment Reset**    | `obs = env.reset()`                               | `obs, info = env.reset()`                                     | Training loops  |
| **Random Seeding**       | `env.seed(42)`                                    | `env.reset(seed=42)`                                          | Reproducibility |
| **Step Function**        | `obs, reward, done, info = env.step(action)`      | `obs, reward, terminated, truncated, info = env.step(action)` | RL algorithms   |
| **Episode Ending**       | `while not done:`                                 | `while not (terminated or truncated):`                        | Training loops  |
| **Render Mode**          | `env.render(mode="human")`                        | `gym.make(env_id, render_mode="human")`                       | Visualization   |
| **Time Limit Detection** | `info.get('TimeLimit.truncated')`                 | `truncated` return value                                      | RL algorithms   |
| **Value Bootstrapping**  | `target = reward + (1-done) * gamma * next_value` | `target = reward + (1-terminated) * gamma * next_value`       | RL correctness  |

## Side-by-Side Code Comparison

### Old v0.21 Code
```python
import gym

# Environment creation and seeding
env = gym.make("LunarLander-v3", options={})
env.seed(123)
observation = env.reset()

# Training loop
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render(mode="human")

env.close()
```

### New v0.26+ Code (Including v1.0.0)
```python
import gymnasium as gym  # Note: 'gymnasium' not 'gym'

# Environment creation with render mode specified upfront
env = gym.make("LunarLander-v3", render_mode="human")

# Reset with seed parameter
observation, info = env.reset(seed=123, options={})

# Training loop with terminated/truncated distinction
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Episode ends if either terminated OR truncated
    done = terminated or truncated

env.close()
```

## Key Changes Breakdown

### 1. Package Name Change

**Old**: `import gym`
**New**: `import gymnasium as gym`

Why: Gymnasium is a separate project that maintains and improves upon the original Gym codebase.

```python
# OLD
import gym

# NEW
import gymnasium as gym
```

### 2. Seeding and Random Number Generation

The biggest conceptual change is how randomness is handled.

**Old v0.21**: Separate `seed()` method
```python
env = gym.make("CartPole-v1")
env.seed(42)  # Set random seed
obs = env.reset()  # Reset environment
```

**New v0.26+**: Seed passed to `reset()`
```python
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)  # Seed and reset together
```

**Why this changed**: Some environments (especially emulated games) can only set their random state at the beginning of an episode, not mid-episode. The old approach could lead to inconsistent behavior.

**Practical impact**:
```python
# OLD: Seeding applied to all future episodes
env.seed(42)
for episode in range(10):
    obs = env.reset()

# NEW: Each episode can have its own seed
for episode in range(10):
    obs, info = env.reset(seed=42 + episode)  # Each episode gets unique seed
```

### 3. Environment Reset Changes

**Old v0.21**: Returns only observation
```python
observation = env.reset()
```

**New v0.26+**: Returns observation AND info
```python
observation, info = env.reset()
```

**Why this changed**:
- `info` provides consistent access to debugging information
- `seed` parameter enables reproducible episodes
- `options` parameter allows episode-specific configuration

**Common migration pattern**:
```python
# If you don't need the new features, just unpack the tuple
obs, _ = env.reset()  # Ignore info with underscore

# If you want to maintain the same random behavior as v0.21
env.reset(seed=42)  # Set seed once
# Then for subsequent resets:
obs, info = env.reset()  # Uses internal random state
```

### 4. Step Function: The `done` → `terminated`/`truncated` Split

This is the most important change for training algorithms.

**Old v0.21**: Single `done` flag
```python
obs, reward, done, info = env.step(action)
```

**New v0.26+**: Separate `terminated` and `truncated` flags
```python
obs, reward, terminated, truncated, info = env.step(action)
```

**Why this matters**:
- **`terminated`**: Episode ended because the task was completed or failed (agent reached goal, died, etc.)
- **`truncated`**: Episode ended due to external constraints (time limit, step limit, etc.)

This distinction is crucial for value function bootstrapping in RL algorithms:

```python
# OLD (ambiguous)
if done:
    # Should we bootstrap? We don't know if this was natural termination or time limit!
    next_value = 0  # Assumption that may be wrong

# NEW (clear)
if terminated:
    next_value = 0      # Natural ending - no future value
elif truncated:
    next_value = value_function(next_obs)  # Time limit - estimate future value
```

**Migration strategy**:
```python
# Simple migration (works for many cases)
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

# Better migration (preserves RL algorithm correctness)
obs, reward, terminated, truncated, info = env.step(action)
if terminated:
    # Episode naturally ended - use reward as-is
    target = reward
elif truncated:
    # Episode cut short - may need to estimate remaining value
    target = reward + discount * estimate_value(obs)
```

For more information, see our [blog post](https://farama.org/Gymnasium-Terminated-Truncated-Step-API) about it.

### 5. Render Mode Changes

**Old v0.21**: Render mode specified each time
```python
env = gym.make("CartPole-v1")
env.render(mode="human")     # Visual window
env.render(mode="rgb_array") # Get pixel array
```

**New v0.26+**: Render mode fixed at creation
```python
env = gym.make("CartPole-v1", render_mode="human")     # For visual display
env = gym.make("CartPole-v1", render_mode="rgb_array") # For recording
env.render()  # Uses the mode specified at creation
```

**Why this changed**: Some environments can't switch render modes on-the-fly. Fixing the mode at creation enables better optimization and prevents bugs.

**Practical implications**:
```python
# OLD: Could switch modes dynamically
env = gym.make("CartPole-v1")
for episode in range(10):
    # ... episode code ...
    if episode % 10 == 0:
        env.render(mode="human")  # Show every 10th episode

# NEW: Create separate environments for different purposes
training_env = gym.make("CartPole-v1")  # No rendering for speed
eval_env = gym.make("CartPole-v1", render_mode="human")  # Visual for evaluation

# Or use None for no rendering, then create visual env when needed
env = gym.make("CartPole-v1", render_mode=None)  # Fast training
if need_visualization:
    visual_env = gym.make("CartPole-v1", render_mode="human")
```

## TimeLimit Wrapper Changes

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

The :class:`TimeLimit` wrapper behavior also changed to align with the new termination model.
```
**Old v0.21**: Added `TimeLimit.truncated` to info dict
```python
obs, reward, done, info = env.step(action)
if done and info.get('TimeLimit.truncated', False):
    # Episode ended due to time limit
    pass
```

**New v0.26+**: Uses the `truncated` return value
```python
obs, reward, terminated, truncated, info = env.step(action)
if truncated:
    # Episode ended due to time limit (or other truncation)
    pass
if terminated:
    # Episode ended naturally (success/failure)
    pass
```

This makes time limit detection much cleaner and more explicit.

## Environment-Specific Changes

### Removed Environments

Some environments were moved or removed:

```python
# OLD: Robotics environments in main gym
import gym
env = gym.make("FetchReach-v1")  # No longer available

# NEW: Moved to separate package
import gymnasium

import gymnasium_robotics
import ale_py

gymnasium.register_envs((gymnasium_robotics, ale_py))

env = gymnasium.make("FetchReach-v1")
env = gymnasium.make("ALE/Pong-v5")
```

## Compatibility Helpers

### Loading OpenAI Gym environments

For environments that can't be updated to Gymnasium, we provide compatibility wrappers either for v21 and v26 style environments, where either the environment name or the environment itself can be passed.

```python
import gymnasium
import shimmy  # install shimmy with `pip install shimmy`

gymnasium.register_envs(shimmy)


# Gym v0.21 style environments
env = gymnasium.make("GymV21Environment-v0", env_id="CartPole-v1")
# or
env = gymnasium.make("GymV21Environment-v0", env=OldV21Env())

# Gym v0.26 style environments
env = gymnasium.make("GymV26Environment-v0", env_id="Cartpole-v1")
# or
env = gymnasium.make("GymV26Environment-v0", env=OldV26Env())
```

### Step API Compatibility

```{eval-rst}
.. py:currentmodule:: gymnasium.utils.step_api_compatibility

If environments implement the (old) done step API, Gymnasium provides functions (:meth:`convert_to_terminated_truncated_step_api` and :meth:`convert_to_done_step_api`) that will convert an environment with the old step API (using ``done``) to the new step API (using ``termination`` and ``truncation``), and vice versa.
```

## Testing Your Migration

After migrating, verify that:

- [ ] **Import statements** use `gymnasium` instead of `gym`
- [ ] **Reset calls** handle the `(obs, info)` return format
- [ ] **Step calls** handle `terminated` and `truncated` separately
- [ ] **Render mode** is specified during environment creation
- [ ] **Random seeding** uses the `seed` parameter in `reset()`
- [ ] **Training algorithms** properly distinguish termination types

```{eval-rst}
.. py:currentmodule:: gymnasium.utils.env_checker

Use the :meth:`check_env` to verify their implementation.
```

## Getting Help

**If you encounter issues during migration**:

1. **Check the compatibility guide**: Some old environments can be used with compatibility wrappers
2. **Look at the environment documentation**: Each environment may have specific migration notes
3. **Test with simple environments first**: Start with CartPole before moving to complex environments
4. **Compare old vs new behavior**: Run the same code with both APIs to understand differences

**Common resources**:
- [Gymnasium API documentation](https://gymnasium.farama.org/api/env)
- [GitHub issues](https://github.com/Farama-Foundation/Gymnasium/issues) for bug reports
- [Discord community](https://discord.gg/bnJ6kubTg6) for questions
