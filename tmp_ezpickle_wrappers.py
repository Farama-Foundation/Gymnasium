import copy
import dataclasses
import importlib
import inspect
import json
import re
from typing import Any, Union

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.envs.registration import EnvSpec, load, SpecStack



def reconstruct_env(stack) -> gym.Env:
    env = gym.make(id=stack[-1], allow_default_wrappers=False)
    for i in range(len(stack) - 1):
        ws = stack[-2 - i]
        if ws.entry_point is None:
            raise gym.error.Error(f"{ws.id} registered but entry_point is not specified")
        elif callable(ws.entry_point):
            env_creator = ws.entry_point
        else:
            # Assume it's a string
            env_creator = load(ws.entry_point)

        env = env_creator(env, *ws.args, **ws.kwargs)

    return env


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

print(SpecStack(gym.make(SpecStack(env))))

print("Done")
