---
layout: "contents"
title: Train an Agent
---

# Training an Agent

When we talk about training an RL agent, we're teaching it to make good decisions through experience. Unlike supervised learning where we show examples of correct answers, RL agents learn by trying different actions and observing the results. It's like learning to ride a bike - you try different movements, fall down a few times, and gradually learn what works.

The goal is to develop a **policy** - a strategy that tells the agent what action to take in each situation to maximize long-term rewards.

## Understanding Q-Learning Intuitively

For this tutorial, we'll use Q-learning to solve the Blackjack environment. But first, let's understand how Q-learning works conceptually.

Q-learning builds a giant "cheat sheet" called a Q-table that tells the agent how good each action is in each situation:

- **Rows** = different situations (states) the agent can encounter
- **Columns** = different actions the agent can take
- **Values** = how good that action is in that situation (expected future reward)

For Blackjack:
- **States**: Your hand value, dealer's showing card, whether you have a usable ace
- **Actions**: Hit (take another card) or Stand (keep current hand)
- **Q-values**: Expected reward for each action in each state

### The Learning Process

1. **Try an action** and see what happens (reward + new state)
2. **Update your cheat sheet**: "That action was better/worse than I thought"
3. **Gradually improve** by trying actions and updating estimates
4. **Balance exploration vs exploitation**: Try new things vs use what you know works

**Why it works**: Over time, good actions get higher Q-values, bad actions get lower Q-values. The agent learns to pick actions with the highest expected rewards.

---

This page provides a short outline of how to train an agent for a Gymnasium environment. We'll use tabular Q-learning to solve Blackjack-v1. For complete tutorials with other environments and algorithms, see {doc}`training tutorials </tutorials/training_agents/index>`. Please read [basic usage](basic_usage) before this page.

## About the Environment: Blackjack

Blackjack is one of the most popular casino card games and is perfect for learning RL because it has:
- **Clear rules**: Get closer to 21 than the dealer without going over
- **Simple observations**: Your hand value, dealer's showing card, usable ace
- **Discrete actions**: Hit (take card) or Stand (keep current hand)
- **Immediate feedback**: Win, lose, or draw after each hand

This version uses infinite deck (cards drawn with replacement), so card counting won't work - the agent must learn optimal basic strategy through trial and error.

**Environment Details**:
- **Observation**: (player_sum, dealer_card, usable_ace)
  - `player_sum`: Current hand value (4-21)
  - `dealer_card`: Dealer's face-up card (1-10)
  - `usable_ace`: Whether player has usable ace (True/False)
- **Actions**: 0 = Stand, 1 = Hit
- **Rewards**: +1 for win, -1 for loss, 0 for draw
- **Episode ends**: When player stands or busts (goes over 21)

## Executing an action

After receiving our first observation from `env.reset()`, we use `env.step(action)` to interact with the environment. This function takes an action and returns five important values:

```python
observation, reward, terminated, truncated, info = env.step(action)
```

- **`observation`**: What the agent sees after taking the action (new game state)
- **`reward`**: Immediate feedback for that action (+1, -1, or 0 in Blackjack)
- **`terminated`**: Whether the episode ended naturally (hand finished)
- **`truncated`**: Whether episode was cut short (time limits - not used in Blackjack)
- **`info`**: Additional debugging information (can usually be ignored)

The key insight is that `reward` tells us how good our *immediate* action was, but the agent needs to learn about *long-term* consequences. Q-learning handles this by estimating the total future reward, not just the immediate reward.

## Building a Q-Learning Agent

Let's build our agent step by step. We need functions for:
1. **Choosing actions** (with exploration vs exploitation)
2. **Learning from experience** (updating Q-values)
3. **Managing exploration** (reducing randomness over time)

### Exploration vs Exploitation

This is a fundamental challenge in RL:
- **Exploration**: Try new actions to learn about the environment
- **Exploitation**: Use current knowledge to get the best rewards

We use **epsilon-greedy** strategy:
- With probability `epsilon`: choose a random action (explore)
- With probability `1-epsilon`: choose the best known action (exploit)

Starting with high epsilon (lots of exploration) and gradually reducing it (more exploitation as we learn) works well in practice.

```python
from collections import defaultdict
import gymnasium as gym
import numpy as np


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
```

### Understanding the Q-Learning Update

The core learning happens in the `update` method. Let's break down the math:

```python
# Current estimate: Q(state, action)
current_q = self.q_values[obs][action]

# What we actually experienced: reward + discounted future value
target = reward + self.discount_factor * max(self.q_values[next_obs])

# How wrong were we?
error = target - current_q

# Update estimate: move toward the target
new_q = current_q + learning_rate * error
```

This is the famous **Bellman equation** in action - it says the value of a state-action pair should equal the immediate reward plus the discounted value of the best next action.

## Training the Agent

Now let's train our agent. The process is:
1. **Reset environment** to start a new episode
2. **Play one complete hand** (episode), choosing actions and learning from each step
3. **Update exploration rate** (reduce epsilon)
4. **Repeat** for many episodes until the agent learns good strategy

```python
# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 100_000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration

# Create environment and agent
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)
```

### The Training Loop

```python
from tqdm import tqdm  # Progress bar

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from this experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()
```

### What to Expect During Training

**Early episodes (0-10,000)**:
- Agent acts mostly randomly (high epsilon)
- Wins about 43% of hands (slightly worse than random due to poor strategy)
- Large learning errors as Q-values are very inaccurate

**Middle episodes (10,000-50,000)**:
- Agent starts finding good strategies
- Win rate improves to 45-48%
- Learning errors decrease as estimates get better

**Later episodes (50,000+)**:
- Agent converges to near-optimal strategy
- Win rate plateaus around 49% (theoretical maximum for this game)
- Small learning errors as Q-values stabilize

## Analyzing Training Results

Let's visualize the training progress:

```python
from matplotlib import pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()
```

### Interpreting the Results

**Reward Plot**: Should show gradual improvement from ~-0.05 (slightly negative) to ~-0.01 (near optimal). Blackjack is a difficult game - even perfect play loses slightly due to the house edge.

**Episode Length**: Should stabilize around 2-3 actions per episode. Very short episodes suggest the agent is standing too early; very long episodes suggest hitting too often.

**Training Error**: Should decrease over time, indicating the agent's predictions are getting more accurate. Large spikes early in training are normal as the agent encounters new situations.

## Common Training Issues and Solutions

### 🚨 **Agent Never Improves**
**Symptoms**: Reward stays constant, large training errors
**Causes**: Learning rate too high/low, poor reward design, bugs in update logic
**Solutions**:
- Try learning rates between 0.001 and 0.1
- Check that rewards are meaningful (-1, 0, +1 for Blackjack)
- Verify Q-table is actually being updated

### 🚨 **Unstable Training**
**Symptoms**: Rewards fluctuate wildly, never converge
**Causes**: Learning rate too high, insufficient exploration
**Solutions**:
- Reduce learning rate (try 0.01 instead of 0.1)
- Ensure minimum exploration (final_epsilon ≥ 0.05)
- Train for more episodes

### 🚨 **Agent Gets Stuck in Poor Strategy**
**Symptoms**: Improvement stops early, suboptimal final performance
**Causes**: Too little exploration, learning rate too low
**Solutions**:
- Increase exploration time (slower epsilon decay)
- Try higher learning rate initially
- Use different exploration strategies (optimistic initialization)

### 🚨 **Learning Too Slow**
**Symptoms**: Agent improves but very gradually
**Causes**: Learning rate too low, too much exploration
**Solutions**:
- Increase learning rate (but watch for instability)
- Faster epsilon decay (less random exploration)
- More focused training on difficult states

## Testing Your Trained Agent

Once training is complete, test your agent's performance:

```python
# Test the trained agent
def test_agent(agent, env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

# Test your agent
test_agent(agent, env)
```

Good Blackjack performance:
- **Win rate**: 42-45% (house edge makes >50% impossible)
- **Average reward**: -0.02 to +0.01
- **Consistency**: Low standard deviation indicates reliable strategy

## Next Steps

Congratulations! You've successfully trained your first RL agent. Here's what to explore next:

1. **Try other environments**: CartPole, MountainCar, LunarLander
2. **Experiment with hyperparameters**: Learning rates, exploration strategies
3. **Implement other algorithms**: SARSA, Expected SARSA, Monte Carlo methods
4. **Add function approximation**: Neural networks for larger state spaces
5. **Create custom environments**: Design your own RL problems

For more information, see:

* [Basic Usage](basic_usage) - Understanding Gymnasium fundamentals
* [Custom Environments](create_custom_env) - Building your own RL problems
* {doc}`Complete Training Tutorials </tutorials/training_agents/index>` - More algorithms and environments
* [Recording Agent Behavior](record_agent) - Saving videos and performance data

The key insight from this tutorial is that RL agents learn through trial and error, gradually building up knowledge about what actions work best in different situations. Q-learning provides a systematic way to learn this knowledge, balancing exploration of new possibilities with exploitation of current knowledge.

Keep experimenting, and remember that RL is as much art as science - finding the right hyperparameters and environment design often requires patience and intuition!
