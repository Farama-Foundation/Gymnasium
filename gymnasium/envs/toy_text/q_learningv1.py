import numpy as np
import gymnasium as gym
# Import your modified Blackjack environment.
# For example, if it’s defined in blackjack_env.py:
# from blackjack import BlackjackEnv
#
# Otherwise, if you have registered it with Gymnasium as "Blackjack-v1":
# env = gym.make("Blackjack-v1", natural=False, sab=False)
#
# In this example we assume you can instantiate it directly:
from blackjack import BlackjackEnv 
import numpy as np
import gymnasium as gym
# Import your modified Blackjack environment.
# For example, if it’s defined in blackjack_env.py:
def state_to_key(state):
    """
    Recursively convert state to a hashable tuple.
    This handles nested lists, tuples, and numpy arrays.
    """
    if isinstance(state, np.ndarray):
        return tuple(state.tolist())
    elif isinstance(state, (list, tuple)):
        return tuple(state_to_key(s) for s in state)
    else:
        return state

def choose_action(state, Q, n_actions, epsilon):
    # Initialize the state in Q if it doesn't exist.
    if state not in Q:
        Q[state] = np.zeros(n_actions)
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])
    

def main():
    # Create the environment.
    # Adjust the arguments (natural, sab, etc.) as needed.
    env = BlackjackEnv(render_mode=None, natural=False, sab=False)

    # Q-learning parameters
    num_episodes = 100000     # Number of episodes for training
    alpha = 0.001             # Learning rate
    gamma = 1.0              # Discount factor (1.0 since the task is episodic)
    epsilon = 0.99           # Exploration rate

    n_actions = env.action_space.n  # e.g., 4 actions: Stick, Hit, Double Down, Split

    # Initialize the Q-table as a dictionary mapping state -> np.array([Q(s, a)])
    Q = {}

    def get_Q(state):
        """
        Returns the Q-values for a given state. Initializes with zeros if state is unseen.
        """
        key = state_to_key(state)
        if key not in Q:
            Q[key] = np.zeros(n_actions)
        return Q[key]

    print("Starting training...")
    # Training loop using Q-learning.
    for episode in range(num_episodes):
        # Reset environment; gymnasium returns (observation, info)
        obs, info = env.reset()
        state = state_to_key(obs)
        done = False

        while not done:
            # Choose an action using epsilon-greedy policy.
            action = choose_action(state, Q, n_actions, epsilon)
            # Step the environment.
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = state_to_key(next_obs)

            # Q-Learning update
            old_value = get_Q(state)[action]
            next_max = np.max(get_Q(next_state))
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            Q[state][action] = new_value

            state = next_state

        # Optionally print progress
        if (episode + 1) % 10000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed.")

    print("Training complete.")

    # Evaluate the learned policy over several episodes.
    num_eval_episodes = 1000
    total_reward = 0.0

    print("Starting evaluation...")
    for _ in range(num_eval_episodes):
        obs, info = env.reset()
        state = state_to_key(obs)
        done = False
        episode_reward = 0.0

        while not done:
            # Always pick the best (greedy) action during evaluation.
            action = np.argmax(get_Q(state))
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = state_to_key(next_obs)
            done = terminated or truncated

        total_reward += episode_reward

    avg_reward = total_reward / num_eval_episodes
    print(f"Average evaluation reward over {num_eval_episodes} episodes: {avg_reward:.3f}")

    env.close()

if __name__ == "__main__":
    main()