import gymnasium
import gym
import time
import numpy as np
num_steps = 1000
batch_size = 16
num_envs = 64

def make_env():
    env = gym.make("ALE/Pong-v5")
    env.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
    env.action_space = gymnasium.spaces.Discrete(6)
    return env


envs = gymnasium.vector.AsyncVectorEnv(
    [make_env for _ in range(num_envs)],
    batch_size=batch_size
)
start_time = time.time()
for i in range(num_steps * (num_envs // batch_size)):
    if i == 0:
        envs.reset_async()
    obs, rewards, terminateds, truncateds, infos = envs.recv()
    # print(obs.shape, infos)
    envs.send(np.random.randint(envs.single_action_space.n, size=batch_size), infos["env_ids"])
end_time = time.time()
print("======Time taken: ", end_time - start_time)


envs = gymnasium.vector.AsyncVectorEnv(
    [make_env for _ in range(num_envs)],
)
start_time = time.time()
for i in range(num_steps):
    if i == 0:
        envs.reset()
    envs.step(np.random.randint(envs.single_action_space.n, size=num_envs))
end_time = time.time()
print("======Time taken: ", end_time - start_time)