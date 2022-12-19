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
