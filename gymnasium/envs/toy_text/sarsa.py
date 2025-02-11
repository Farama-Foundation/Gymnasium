import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from blackjack import BlackjackEnv  # ensure this file is accessible

def state_to_key(state):
    if isinstance(state, np.ndarray):
        return tuple(state.tolist())
    elif isinstance(state, (list, tuple)):
        return tuple(state_to_key(s) for s in state)
    else:
        return state

def choose_action(state, Q, n_actions, epsilon):
    if state not in Q:
        Q[state] = np.zeros(n_actions)
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

def evaluate_policy(env, Q, n_actions, num_eval_episodes=1000):
    win_count = 0
    total_reward = 0.0
    for _ in range(num_eval_episodes):
        obs, info = env.reset()
        state = state_to_key(obs)
        done = False
        episode_reward = 0.0
        while not done:
            # Use greedy policy for evaluation.
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            action = np.argmax(Q[state])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = state_to_key(next_obs)
            done = terminated or truncated
        total_reward += episode_reward
        if episode_reward > 0.0:
            win_count += 1
    avg_reward = total_reward / num_eval_episodes
    win_rate = win_count / num_eval_episodes * 100
    return avg_reward, win_rate

def main():
    env = BlackjackEnv(render_mode=None, natural=False, sab=False)
    num_episodes = 900000
    alpha = 0.01
    gamma = 1.0
    epsilon = 0.9
    n_actions = env.action_space.n
    Q = {}

    # Lists to store periodic evaluation results.
    evaluation_points = []
    win_rates = []
    avg_rewards = []

    eval_interval = 10000  # evaluate every 10,000 episodes

    print("Starting SARSA training...")
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = state_to_key(obs)
        action = choose_action(state, Q, n_actions, epsilon)
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = state_to_key(next_obs)
            next_action = choose_action(next_state, Q, n_actions, epsilon) if not done else None
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            if not done:
                target = reward + gamma * Q[next_state][next_action]
            else:
                target = reward
            Q[state][action] += alpha * (target - Q[state][action])
            state = next_state
            action = next_action

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            avg_reward, win_rate = evaluate_policy(env, Q, n_actions, num_eval_episodes=1000)
            evaluation_points.append(episode + 1)
            avg_rewards.append(avg_reward)
            win_rates.append(win_rate)
            print(f"Episode {episode + 1}: Average Reward = {avg_reward:.3f}, Win Rate = {win_rate:.2f}%")

    print("Training complete.\n")

    # Final evaluation (if desired)
    num_eval_episodes = 10000
    total_reward = 0.0
    win_count = 0
    bust_count = 0
    push_count = 0
    loss_count = 0
    eval_rewards = []

    print("Starting final evaluation of the learned policy...")
    for _ in range(num_eval_episodes):
        obs, info = env.reset()
        state = state_to_key(obs)
        done = False
        episode_reward = 0.0
        while not done:
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            action = np.argmax(Q[state])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = state_to_key(next_obs)
            done = terminated or truncated
        total_reward += episode_reward
        eval_rewards.append(episode_reward)
        if env.is_bust(env.current_hand):
            bust_count += 1
        elif episode_reward > 0.0:
            win_count += 1
        elif episode_reward == 0.0:
            push_count += 1
        elif episode_reward < 0.0:
            loss_count += 1

    avg_reward_final = total_reward / num_eval_episodes
    print(f"Final Evaluation Average Reward: {avg_reward_final:.3f}")
    print("Outcomes:")
    print(f"  Win Rate   : {win_count / num_eval_episodes * 100:.3f}%")
    print(f"  Bust Rate  : {bust_count / num_eval_episodes * 100:.3f}%")
    print(f"  Loss Rate  : {loss_count / num_eval_episodes * 100:.3f}%")
    print(f"  Push Rate  : {push_count / num_eval_episodes * 100:.3f}%")

    # ------------------- Plotting the Results -------------------
    # Plot win rate over time during training.
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(evaluation_points, win_rates, marker='o')
    plt.title("Win Rate Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (%)")
    plt.grid(True)

    # Plot average reward over time during training.
    plt.subplot(1, 2, 2)
    plt.plot(evaluation_points, avg_rewards, marker='o', color='orange')
    plt.title("Average Reward Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
