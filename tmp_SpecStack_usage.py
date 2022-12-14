import json

import gymnasium as gym
from gymnasium.envs.registration import SpecStack


# construct the environment
env = gym.make("LunarLander-v2")
env = gym.wrappers.TimeAwareObservation(env)
env = gym.wrappers.TransformReward(env, lambda r: 0.01 * r)
env = gym.wrappers.ResizeObservation(env, (84, 84))

# Printing the spec stack
env_spec_stack = env.spec_stack
print(env_spec_stack)

# Reconstructing the environment from the spec stack
reconstructed_env = gym.make(env_spec_stack)

# spec stack as JSON
json_representation = reconstructed_env.spec_stack.json
reconstructed_env_from_json = gym.make(SpecStack(json_representation))      # spec_stack can be used to construct the environment

# The two spec stacks should be identical
print(reconstructed_env_from_json.spec_stack)
print(reconstructed_env_from_json.spec_stack == env_spec_stack)
