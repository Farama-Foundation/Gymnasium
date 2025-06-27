"""
Action Masking in the Taxi Environment
====================================

This tutorial demonstrates how to use action masking in the Taxi environment to improve
reinforcement learning performance by preventing invalid actions.

The Taxi environment is a grid world where a taxi needs to pick up a passenger and drop
them off at their destination. The environment provides an action mask that indicates
which actions are valid in the current state, helping the agent avoid invalid moves
like driving into walls or attempting to pick up/drop off passengers when not in the correct location.

In this tutorial, we will:
1. Create the Taxi environment and understand its action space
2. Examine the action mask functionality
3. Train a Q-learning agent with and without action masking
4. Compare the performance of both approaches
"""

import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym


# Base random seed for reproducibility
BASE_RANDOM_SEED = 58922320

# The action space is discrete with 6 possible actions:
# 0: Move south (down)
# 1: Move north (up)
# 2: Move east (right)
# 3: Move west (left)
# 4: Pickup passenger
# 5: Drop off passenger


def get_action_mask(env, state: int) -> np.ndarray:
    """Get the action mask for a given state."""
    return env.action_mask(state)


def train_q_learning(
    env,
    use_action_mask: bool = True,
    episodes: int = 5000,
    seed: int = BASE_RANDOM_SEED,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 0.1,
) -> Dict[str, float]:
    """Train a Q-learning agent with or without action masking."""
    # Set random seeds for this run
    np.random.seed(seed)
    random.seed(seed)

    # Initialize Q-table
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    # Training metrics
    episode_rewards = []

    for episode in range(episodes):
        # Reset environment with specific seed for this episode
        state, info = env.reset(seed=seed + episode)
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Get action mask if using it
            action_mask = info["action_mask"] if use_action_mask else None

            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Random action (only from valid actions if using mask)
                if use_action_mask:
                    valid_actions = np.where(action_mask == 1)[0]
                    action = np.random.choice(valid_actions)
                else:
                    action = np.random.randint(0, n_actions)
            else:
                # Best action (only from valid actions if using mask)
                if use_action_mask:
                    valid_actions = np.where(action_mask == 1)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[np.argmax(q_table[state, valid_actions])]
                    else:
                        action = np.random.randint(0, n_actions)
                else:
                    action = np.argmax(q_table[state])

            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Q-learning update
            if not (done or truncated):
                if use_action_mask:
                    next_mask = info["action_mask"]
                    valid_next_actions = np.where(next_mask == 1)[0]
                    if len(valid_next_actions) > 0:
                        next_max = np.max(q_table[next_state, valid_next_actions])
                    else:
                        next_max = 0
                else:
                    next_max = np.max(q_table[next_state])

                q_table[state, action] = q_table[state, action] + learning_rate * (
                    reward + discount_factor * next_max - q_table[state, action]
                )

            state = next_state

        episode_rewards.append(total_reward)

    return {
        "q_table": q_table,
        "episode_rewards": episode_rewards,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
    }


def run_experiment(
    seed: int,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 0.1,
    episodes: int = 5000,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run a single experiment with both masked and unmasked agents."""
    # Train agent with action masking
    env_masked = gym.make("Taxi-v3")
    env_masked.reset(seed=seed)
    masked_results = train_q_learning(
        env_masked,
        use_action_mask=True,
        seed=seed,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        episodes=episodes,
    )
    env_masked.close()

    # Train agent without action masking
    env_unmasked = gym.make("Taxi-v3")
    env_unmasked.reset(seed=seed)
    unmasked_results = train_q_learning(
        env_unmasked,
        use_action_mask=False,
        seed=seed,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
    )
    env_unmasked.close()

    return masked_results, unmasked_results


def experiment_qlearning_with_and_without_action_masking(
    n_runs: int = 5,
    episodes: int = 5000,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 0.1,
    savefig_folder: Path = Path(__file__).resolve().parents[2]
    / "_static/img/tutorials/",
):
    """Run multiple experiments comparing Q-learning with and without action masking."""
    # Generate different seeds for each run
    seeds = [BASE_RANDOM_SEED + i for i in range(n_runs)]

    # Run experiments
    masked_results_list = []
    unmasked_results_list = []

    for i, seed in enumerate(seeds):
        print(f"\nRun {i + 1}/{n_runs} with seed {seed}")
        masked_results, unmasked_results = run_experiment(
            seed,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            episodes=episodes,
        )
        masked_results_list.append(masked_results)
        unmasked_results_list.append(unmasked_results)

    # Calculate statistics across runs
    masked_mean_rewards = [r["mean_reward"] for r in masked_results_list]
    unmasked_mean_rewards = [r["mean_reward"] for r in unmasked_results_list]

    masked_mean = np.mean(masked_mean_rewards)
    masked_std = np.std(masked_mean_rewards)
    unmasked_mean = np.mean(unmasked_mean_rewards)
    unmasked_std = np.std(unmasked_mean_rewards)

    # Plot results for all runs
    plt.figure(figsize=(10, 6), dpi=100)

    # Plot individual runs
    for i, (masked_results, unmasked_results) in enumerate(
        zip(masked_results_list, unmasked_results_list)
    ):
        plt.plot(
            masked_results["episode_rewards"],
            label="With Action Masking" if i == 0 else None,
            color="blue",
            alpha=0.05,
        )
        plt.plot(
            unmasked_results["episode_rewards"],
            label="Without Action Masking" if i == 0 else None,
            color="red",
            alpha=0.05,
        )

    # Calculate and plot mean across runs
    masked_mean_curve = np.mean(
        [r["episode_rewards"] for r in masked_results_list], axis=0
    )
    unmasked_mean_curve = np.mean(
        [r["episode_rewards"] for r in unmasked_results_list], axis=0
    )

    plt.plot(
        masked_mean_curve, label="With Action Masking (Mean)", color="blue", linewidth=1
    )
    plt.plot(
        unmasked_mean_curve,
        label="Without Action Masking (Mean)",
        color="red",
        linewidth=1,
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance: With vs Without Action Masking")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        savefig_folder / "taxi_v3_action_masking_comparison.png", bbox_inches="tight"
    )
    plt.close()

    print("\nResults across all runs:")
    print("With Action Masking:")
    print(f"  Mean Reward: {masked_mean:.2f} ± {masked_std:.2f}")
    print(f"  Individual run means: {[float(f'{r:.2f}') for r in masked_mean_rewards]}")
    print("Without Action Masking:")
    print(f"  Mean Reward: {unmasked_mean:.2f} ± {unmasked_std:.2f}")
    print(
        f"  Individual run means: {[float(f'{r:.2f}') for r in unmasked_mean_rewards]}"
    )


if __name__ == "__main__":
    experiment_qlearning_with_and_without_action_masking(
        n_runs=12,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,
        episodes=5000,
        savefig_folder=Path(__file__).resolve().parents[2] / "_static/img/tutorials/",
    )
