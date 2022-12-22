import json

import gymnasium as gym
from gymnasium.utils.serialize_spec_stack import serialise_spec_stack, deserialise_spec_stack, pprint_stack


# construct the environment
env = gym.make("LunarLander-v2")
env = gym.wrappers.TimeAwareObservation(env)
env = gym.wrappers.TransformReward(env, lambda r: 0.01 * r)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.TransformObservation(env, lambda o: o / 255.0)

# encoding process
stack = env.spec_stack                                     # spec stack
as_json = serialise_spec_stack(stack)                      # serialise to JSON
as_string = json.dumps(as_json)                            # serialise to string

# decoding process
as_json_r = json.loads(as_string)                          # deserialise from string
stack_r = deserialise_spec_stack(as_json_r, eval_ok=True)  # deserialise from JSON
env_r = gym.make(stack_r)                                  # reconstruct the environment

# visualise the spec stack of the reconstructed environment
pprint_stack(as_json_r)
pprint_stack(serialise_spec_stack(env_r.spec_stack))  # NB: This fails, reserialisation of callable is not supported. (env_r.spec_stack is called twice, once in gym.make() and once here)
                                                        # To fix this, env_r.spec_stack needs to be overwritten with stack_r.
                                                        # I don't know how to do this whilst keeping spec_stack as a property.
                                                        # I don't think this would be an issue in practice, as there is no need to call env.spec_stack twice - users can save the output of env.spec_stack as a variable.
                                                        # This only occurs when callables are in the spec stack.
