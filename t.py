import gymnasium
import time
import numpy as np
num_steps = 100
batch_size = 16
num_envs = 64


envs = gymnasium.vector.AsyncVectorEnv(
    [lambda: gymnasium.make("CartPole-v1") for _ in range(num_envs)],
    batch_size=batch_size
)
start_time = time.time()
for i in range(num_steps * (num_envs // batch_size)):
    if i == 0:
        envs.reset_async()
    obs, rewards, terminateds, truncateds, infos = envs.recv()
    print(obs, infos)
    raise
    envs.send(np.random.randint(envs.single_action_space.n, size=batch_size), infos["env_ids"])
end_time = time.time()
print("Time taken: ", end_time - start_time)


envs = gymnasium.vector.AsyncVectorEnv(
    [lambda: gymnasium.make("CartPole-v1") for _ in range(num_envs)],
)
start_time = time.time()
for i in range(num_steps):
    if i == 0:
        envs.reset()
    envs.step(np.random.randint(envs.single_action_space.n, size=num_envs))
end_time = time.time()
print("Time taken: ", end_time - start_time)