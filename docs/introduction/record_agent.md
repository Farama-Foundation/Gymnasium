---
layout: "contents"
title: Recording Agents
---

# Recording Agents

## Why Record Your Agent?

Recording agent behavior serves several important purposes in RL development:

**🎥 Visual Understanding**: See exactly what your agent is doing - sometimes a 10-second video reveals issues that hours of staring at reward plots miss.

**📊 Performance Tracking**: Collect systematic data about episode rewards, lengths, and timing to understand training progress.

**🐛 Debugging**: Identify specific failure modes, unusual behaviors, or environments where your agent struggles.

**📈 Evaluation**: Compare different training runs, algorithms, or hyperparameters objectively.

**🎓 Communication**: Share results with collaborators, include in papers, or create educational content.

## When to Record

**During Evaluation** (Record Every Episode):
- Testing a trained agent's final performance
- Creating demonstration videos
- Detailed analysis of specific behaviors

**During Training** (Record Periodically):
- Monitor learning progress over time
- Catch training issues early
- Create timelapse videos of learning

```{eval-rst}
.. py:currentmodule: gymnasium.wrappers

Gymnasium provides two essential wrappers for recording: :class:`RecordEpisodeStatistics` for numerical data and :class:`RecordVideo` for visual recordings. The first tracks episode metrics like total rewards, episode length, and time taken. The second generates MP4 videos of agent behavior using environment renderings.

We'll show how to use these wrappers for two common scenarios: recording data for every episode (typically during evaluation) and recording data periodically (during training).
```

## Recording Every Episode (Evaluation)

```{eval-rst}
.. py:currentmodule: gymnasium.wrappers

When evaluating a trained agent, you typically want to record several episodes to understand average performance and consistency. Here's how to set this up with :class:`RecordEpisodeStatistics` and :class:`RecordVideo`.
```

```python
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np

# Configuration
num_eval_episodes = 4
env_name = "CartPole-v1"  # Replace with your environment

# Create environment with recording capabilities
env = gym.make(env_name, render_mode="rgb_array")  # rgb_array needed for video recording

# Add video recording for every episode
env = RecordVideo(
    env,
    video_folder="cartpole-agent",    # Folder to save videos
    name_prefix="eval",               # Prefix for video filenames
    episode_trigger=lambda x: True    # Record every episode
)

# Add episode statistics tracking
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

print(f"Starting evaluation for {num_eval_episodes} episodes...")
print(f"Videos will be saved to: cartpole-agent/")

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0

    episode_over = False
    while not episode_over:
        # Replace this with your trained agent's policy
        action = env.action_space.sample()  # Random policy for demonstration

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        episode_over = terminated or truncated

    print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}")

env.close()

# Print summary statistics
print(f'\nEvaluation Summary:')
print(f'Episode durations: {list(env.time_queue)}')
print(f'Episode rewards: {list(env.return_queue)}')
print(f'Episode lengths: {list(env.length_queue)}')

# Calculate some useful metrics
avg_reward = np.sum(env.return_queue)
avg_length = np.sum(env.length_queue)
std_reward = np.std(env.return_queue)

print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
print(f'Average episode length: {avg_length:.1f} steps')
print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')
```

### Understanding the Output

After running this code, you'll find:

**Video Files**: `cartpole-agent/eval-episode-0.mp4`, `eval-episode-1.mp4`, etc.
- Each file shows one complete episode from start to finish
- Useful for seeing exactly how your agent behaves
- Can be shared, embedded in presentations, or analyzed frame-by-frame

**Console Output**: Episode-by-episode performance plus summary statistics
```
Episode 1: 23 steps, reward = 23.0
Episode 2: 15 steps, reward = 15.0
Episode 3: 200 steps, reward = 200.0
Episode 4: 67 steps, reward = 67.0

Average reward: 76.25 ± 78.29
Average episode length: 76.2 steps
Success rate: 100.0%
```

**Statistics Queues**: Time, reward, and length data for each episode
- `env.time_queue`: How long each episode took (wall-clock time)
- `env.return_queue`: Total reward for each episode
- `env.length_queue`: Number of steps in each episode

```{eval-rst}
.. py:currentmodule: gymnasium.wrappers

In the script above, the :class:`RecordVideo` wrapper saves videos with filenames like "eval-episode-0.mp4" in the specified folder. The ``episode_trigger=lambda x: True`` ensures every episode is recorded.

The :class:`RecordEpisodeStatistics` wrapper tracks performance metrics in internal queues that we access after evaluation to compute averages and other statistics.

For computational efficiency during evaluation, it's possible to implement this with vector environments to evaluate N episodes in parallel rather than sequentially.
```

## Recording During Training (Periodic)

During training, you'll run hundreds or thousands of episodes, so recording every one isn't practical. Instead, record periodically to track learning progress:

```python
import logging
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# Training configuration
training_period = 250           # Record video every 250 episodes
num_training_episodes = 10_000  # Total training episodes
env_name = "CartPole-v1"

# Set up logging for episode statistics
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Create environment with periodic video recording
env = gym.make(env_name, render_mode="rgb_array")

# Record videos periodically (every 250 episodes)
env = RecordVideo(
    env,
    video_folder="cartpole-training",
    name_prefix="training",
    episode_trigger=lambda x: x % training_period == 0  # Only record every 250th episode
)

# Track statistics for every episode (lightweight)
env = RecordEpisodeStatistics(env)

print(f"Starting training for {num_training_episodes} episodes")
print(f"Videos will be recorded every {training_period} episodes")
print(f"Videos saved to: cartpole-training/")

for episode_num in range(num_training_episodes):
    obs, info = env.reset()
    episode_over = False

    while not episode_over:
        # Replace with your actual training agent
        action = env.action_space.sample()  # Random policy for demonstration
        obs, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

    # Log episode statistics (available in info after episode ends)
    if "episode" in info:
        episode_data = info["episode"]
        logging.info(f"Episode {episode_num}: "
                    f"reward={episode_data['r']:.1f}, "
                    f"length={episode_data['l']}, "
                    f"time={episode_data['t']:.2f}s")

        # Additional analysis for milestone episodes
        if episode_num % 1000 == 0:
            # Look at recent performance (last 100 episodes)
            recent_rewards = list(env.return_queue)[-100:]
            if recent_rewards:
                avg_recent = sum(recent_rewards) / len(recent_rewards)
                print(f"  -> Average reward over last 100 episodes: {avg_recent:.1f}")

env.close()
```

### Training Recording Benefits

**Progress Videos**: Watch your agent improve over time
- `training-episode-0.mp4`: Random initial behavior
- `training-episode-250.mp4`: Some patterns emerging
- `training-episode-500.mp4`: Clear improvement
- `training-episode-1000.mp4`: Competent performance

**Learning Curves**: Plot episode statistics over time
```python
import matplotlib.pyplot as plt

# Plot learning progress
episodes = range(len(env.return_queue))
rewards = list(env.return_queue)

plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, alpha=0.3, label='Episode Rewards')

# Add moving average for clearer trend
window = 100
if len(rewards) > window:
    moving_avg = [sum(rewards[i:i+window])/window
                  for i in range(len(rewards)-window+1)]
    plt.plot(range(window-1, len(rewards)), moving_avg,
             label=f'{window}-Episode Moving Average', linewidth=2)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Integration with Experiment Tracking

For more sophisticated projects, integrate with experiment tracking tools:

```python
# Example with Weights & Biases (wandb)
import wandb

# Initialize experiment tracking
wandb.init(project="cartpole-training", name="q-learning-run-1")

# Log episode statistics
for episode_num in range(num_training_episodes):
    # ... training code ...

    if "episode" in info:
        episode_data = info["episode"]
        wandb.log({
            "episode": episode_num,
            "reward": episode_data['r'],
            "length": episode_data['l'],
            "episode_time": episode_data['t']
        })

        # Upload videos periodically
        if episode_num % training_period == 0:
            video_path = f"cartpole-training/training-episode-{episode_num}.mp4"
            if os.path.exists(video_path):
                wandb.log({"training_video": wandb.Video(video_path)})
```

## Best Practices Summary

**For Evaluation**:
- Record every episode to get complete performance picture
- Use multiple seeds for statistical significance
- Save both videos and numerical data
- Calculate confidence intervals for metrics

**For Training**:
- Record periodically (every 100-1000 episodes)
- Focus on episode statistics over videos during training
- Use adaptive recording triggers for interesting episodes
- Monitor memory usage for long training runs

**For Analysis**:
- Create moving averages to smooth noisy learning curves
- Look for patterns in both success and failure episodes
- Compare agent behavior at different stages of training
- Save raw data for later analysis and comparison

## More Information

* [Training an agent](train_agent) - Learn how to build the agents you're recording
* [Basic usage](basic_usage) - Understand Gymnasium fundamentals
* {doc}`More training tutorials </tutorials/training_agents/index>` - Advanced training techniques
* [Custom environments](create_custom_env) - Create your own environments to record

Recording agent behavior is an essential skill for RL practitioners. It helps you understand what your agent is actually learning, debug training issues, and communicate results effectively. Start with simple recording setups and gradually add more sophisticated analysis as your projects grow in complexity!
