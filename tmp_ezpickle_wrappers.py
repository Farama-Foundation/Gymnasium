import dataclasses
import importlib
import json
from typing import Any, Union

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.envs.registration import EnvSpec


@dataclasses.dataclass
class WrapperSpec:
    name: str
    entry_point: str
    args: list[Any]
    kwargs: list[Any]


def spec_stack(self) -> tuple[Union[WrapperSpec, EnvSpec]]:
    wrapper_spec = WrapperSpec(type(self).__name__, self.__module__ + ":" + type(self).__name__, self._ezpickle_args, self._ezpickle_kwargs)
    if isinstance(self.env, Wrapper):
         return (wrapper_spec,) + spec_stack(self.env)
    else:
         return (wrapper_spec,) + (self.env.spec,)


def serialise_spec_stack(stack: tuple[Union[WrapperSpec, EnvSpec]]) -> str:
    num_layers = len(stack)
    stack_json = {}
    for i, spec in enumerate(stack):
        if i == num_layers - 1:
            layer = "raw_env"
        else:
            layer = f"wrapper_{num_layers - i - 2}"
        spec_json = json.dumps(dataclasses.asdict(spec))
        stack_json[layer] = spec_json
    return stack_json


def deserialise_spec_stack(stack_json: str) -> tuple[Union[WrapperSpec, EnvSpec]]:
    stack = []
    for name, spec_json in stack_json.items():
        spec = json.loads(spec_json)

        if name != "raw_env":  # EnvSpecs do not have args, o   nly kwargs
            for k, v in enumerate(spec['args']):  # json saves tuples as lists, so we need to convert them back (assumes depth <2, todo: recursify this)
                if type(v) == list:
                    for i, x in enumerate(v):
                        if type(x) == list:
                            spec['args'][k][i] = tuple(x)
                    spec['args'][k] = tuple(v)
            spec['args'] = tuple(spec['args'])

        for k, v in spec['kwargs'].items():  # json saves tuples as lists, so we need to convert them back (assumes depth <2, todo: recursify this)
            if type(v) == list:
                for i, x in enumerate(v):
                    if type(x) == list:
                        spec['kwargs'][k][i] = tuple(x)
                spec['kwargs'][k] = tuple(v)

        if name == "raw_env":
            for key in ['namespace', 'name', 'version']:  # remove args where init is set to False
                spec.pop(key)
            spec = EnvSpec(**spec)
        else:
            spec = WrapperSpec(**spec)
        stack.append(spec)

    return tuple(stack)
    #WrapperSpec(*list(json.loads(stack_json['wrapper_4'].replace("\'", "\"")).values()))


def load(name: str) -> callable:
    """COPIED FROM registration.py
    Loads an environment with name and returns an environment creation function

    Args:
        name: The environment name

    Returns:
        Calls the environment constructor
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def reconstruct_env(stack: tuple[Union[WrapperSpec, EnvSpec]]) -> gym.Env:
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
#env = gym.wrappers.TransformReward(env, lambda r: 0.01 * r)
env = gym.wrappers.ResizeObservation(env, (84, 84))

# get the spec stack
stack = spec_stack(env)

# jsonise the spec stack
serialised_stack = serialise_spec_stack(stack)

# deserialise the spec stack
deserialised_stack = deserialise_spec_stack(serialised_stack)
assert deserialised_stack == stack

# reconstruct the environment
reconstructed_env = reconstruct_env(deserialised_stack)
assert spec_stack(reconstructed_env) == spec_stack(env)

print("Done")

# todo: make ezpickle calls into kwargs not args
