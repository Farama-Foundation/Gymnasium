import gymnasium as gym
import time

env = gym.make('A1Soccer', render_mode = 'human')
env.reset()


# env.print_obs()
# while True:
#     env.render()
#     # print("OBSERVATION COMING UP")
#     # print("OBSERVATION: ", env._get_obs())


while True:  # run for 100 steps or adjust as needed
    env.step(env.action_space.sample())

    time.sleep(0.25)
    print("OBSERVATION COMING UP")
    env.print_obs()
      # you can use a random action or some default action
    env.render()