import gymnasium as gym


def run_game(name: str) -> None:
    try:
        env = gym.make(name, render_mode="human")
        _, _ = env.reset()

        for _ in range(1000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                _, _ = env.reset()

        env.close()
    except KeyboardInterrupt:
        pass
