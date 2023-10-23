import gymnasium as gym
import time

env = gym.make('A1Soccer', render_mode = 'human')
env.reset()

while True:
    env.render()