import json

import gymnasium as gym
from gymnasium.utils.serialize_spec_stack import serialise_spec_stack, deserialise_spec_stack, pprint_stack


# construct the environment
env = gym.make("LunarLander-v2")
env = gym.wrappers.TimeAwareObservation(env)
env = gym.wrappers.TransformReward(env, lambda r: 0.01 * r)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.TransformObservation(env, lambda o: o / 255.0)


print(env.spec_stack)
as_json = serialise_spec_stack(env.spec_stack)
print(as_json)
reconstructed = deserialise_spec_stack(as_json, eval_ok=True)
print(reconstructed)
pprint_stack(as_json)

# # Printing the spec stack
# env_spec_stack = env.spec_stack
# print(env_spec_stack)
#
# # Reconstructing the environment from the spec stack
# reconstructed_env = gym.make(env_spec_stack)
#
# # spec stack as JSON
# json_representation = reconstructed_env.spec_stack.json
# reconstructed_env_from_json = gym.make(SpecStack(json_representation))      # spec_stack can be used to construct the environment
#
# # The two spec stacks should be identical
# print(reconstructed_env_from_json.spec_stack)
# print(reconstructed_env_from_json.spec_stack == env_spec_stack)
