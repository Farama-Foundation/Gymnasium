import json

import gymnasium as gym
from gymnasium.envs.registration import SpecStack

# construct the environment
env = gym.make("CartPole-v1")
env = gym.wrappers.TimeAwareObservation(env)
env = gym.wrappers.TransformReward(env, lambda r: 0.01 * r)
env = gym.wrappers.ResizeObservation(env, (84, 84))

# Example 1: Generate a spec stack from a [wrapped] environment
stack_from_env = SpecStack(env)             # generation is an easy one-liner
print(stack_from_env)                       # string representation is a nice table    todo: make callables readable
readable_stack = stack_from_env.stack_json  # serialise the stack to a readable json string
gym.make(stack_from_env)

# Example 2: Generate a spec stack from a dict representation
stack_from_dict = SpecStack(stack_from_env.stack_json)
print(stack_from_dict)
readable_stack = stack_from_dict.stack_json
gym.make(stack_from_dict)

# Show equality
print(SpecStack(gym.make(SpecStack(env))) == stack_from_env)  # True
env = gym.wrappers.TransformObservation(env, lambda r: 0.01 * r)
print(SpecStack(gym.make(SpecStack(env))) == stack_from_env)  # False

# Example 3: Generate a spec stack from a json string
json.dump(readable_stack, open("tmp_SpecStack_usage.json", "w"))
stack_from_json = SpecStack(json.load(open("tmp_SpecStack_usage.json", "r")))
gym.make(stack_from_json)
print(stack_from_json)
print(stack_from_json == stack_from_dict)  # True
