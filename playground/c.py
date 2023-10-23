import gymnasium as gym
import time

env = gym.make('HalfCheetah-v5', render_mode = 'human')
env.reset()

while True:
    env.render()



# height, width, channels = env.observation_space.shape
# actions = env.action_space.n
# env.unwrapped.get_action_meanings()
# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 
# while not done:
#     time.sleep(10)
#     action = random.choice([0,1,2,3,4,5])
#     n_state, reward, done, info = env.step(action)       
#     score+=reward
# print('Episode:{} Score:{}'.format(episode, score))
# env.close()