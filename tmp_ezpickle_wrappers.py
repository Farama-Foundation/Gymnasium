import dataclasses
import importlib
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
    wrapper_spec = WrapperSpec(type(self).__name__, self.__module__ + ":" + type(self).__name__, self._ezpickle_args, self._ezpickle_kwargs)
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

def load(name: str) -> callable:
    """Loads an environment with name and returns an environment creation function

    Args:
        name: The environment name

    Returns:
        Calls the environment constructor
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

del env
env = gym.make(id=stack[-1], allow_default_wrappers=False)
for i in range(num_layers - 1):
    ws = stack[-2 - i]
    if ws.entry_point is None:
        raise gym.error.Error(f"{ws.id} registered but entry_point is not specified")
    elif callable(ws.entry_point):
        env_creator = ws.entry_point
    else:
        # Assume it's a string
        env_creator = load(ws.entry_point)

    print(f"Creating wrapper {ws.name} with args {ws.args} and kwargs {ws.kwargs}")


# todo: when calling gym.make from a stack, turn all default wrappers off

# todo: make ezpickle calls into kwargs not args
