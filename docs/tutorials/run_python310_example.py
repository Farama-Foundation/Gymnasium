"""
Demonstrates that Gymnasium code examples require Python 3.10+ and the latest Gymnasium version.
"""

import gymnasium as gym


env = gym.make("CartPole-v1")
observation, info = env.reset(seed=123)

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
