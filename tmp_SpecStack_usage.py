import json

import gymnasium as gym
from gymnasium.envs.registration import SpecStack


# construct the environment
env = gym.make("LunarLander-v2")
env = gym.wrappers.TimeAwareObservation(env)
env = gym.wrappers.TransformReward(env, lambda r: 0.01 * r)
env = gym.wrappers.ResizeObservation(env, (84, 84))

# Example 1: Generate a spec stack from a [wrapped] environment
#stack_from_env = SpecStack(env)  # generation is an easy one-liner
print(env.spec_stack)  # string representation is a nice table
json_representation = env.spec_stack.stack_json  # serialise the stack to a readable json string
gym.make(env.spec_stack)

# Example 2: Generate a spec stack from a dict representation
stack_from_dict = SpecStack(json_representation)
print(stack_from_dict)
envtwo = gym.make(stack_from_dict)
print(envtwo.spec_stack)

# # Show equality
# print(SpecStack(gym.make(SpecStack(env))) == stack_from_env)  # True
# env = gym.wrappers.TransformObservation(env, lambda r: 0.01 * r)
# print(SpecStack(gym.make(SpecStack(env))) == stack_from_env)  # False
#
# # Example 3: Generate a spec stack from a json string
# json.dump(readable_stack, open("tmp_SpecStack_usage.json", "w"))
# stack_from_json = SpecStack(json.load(open("tmp_SpecStack_usage.json")))
# gym.make(stack_from_json)
# print(stack_from_json)
# print(stack_from_json == stack_from_dict)  # True
# print(env.spec_stack)
