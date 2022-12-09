import dataclasses
import json
from typing import Any

import gymnasium as gym
from gymnasium import Wrapper


@dataclasses.dataclass
class WrapperSpec:
    name: str
    entry_point: str
    args: list[Any]
    kwargs: list[Any]


def spec_stack(self):
    wrapper_spec = WrapperSpec(type(self).__name__, self.__module__ + "." + type(self).__name__, self._ezpickle_args, self._ezpickle_kwargs)
    if isinstance(self.env, Wrapper):
         return (wrapper_spec,) + spec_stack(self.env)
    else:
         return (wrapper_spec,) + (self.env.spec,)


env = gym.make("CartPole-v1")
env = gym.wrappers.TimeAwareObservation(env)
#env = gym.wrappers.TransformReward(env, lambda r: 0.01 * r)
env = gym.wrappers.ResizeObservation(env, (84, 84))

stack = spec_stack(env)

num_layers = len(stack)
stack_json = {}
for i, spec in enumerate(stack):
    if i == num_layers - 1:
        layer = "raw_env"
    else:
        layer = f"wrapper_{num_layers - i - 2}"
    spec_json = json.dumps(dataclasses.asdict(spec))
    stack_json[layer] = spec_json
print(stack_json)

# todo: when calling gym.make from a stack, turn all default wrappers off

# todo: make ezpickle calls into kwargs not args
