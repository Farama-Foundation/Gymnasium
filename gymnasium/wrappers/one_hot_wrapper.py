import gymnasium as gym
import numpy as np

class OneHotV0(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, task_idx: int, num_envs: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        #self.env = env
        assert task_idx < num_envs, "The task idx of an env cannot be greater than or equal to the number of envs"
        self.one_hot = np.zeros(num_envs)
        self.one_hot[task_idx] = 1

    def step(self, action):
        next_state, reward, terminate, truncate, info = self.env.step(action)
        next_state = np.concatenate([next_state, self.one_hot])
        return next_state, reward, terminate, truncate, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs = np.concatenate([obs, self.one_hot])
        return obs, info
