import copy
import dataclasses
import importlib
import inspect
import json
import re
from typing import Any, Union

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.envs.registration import EnvSpec, load


@dataclasses.dataclass
class WrapperSpec:
    name: str
    entry_point: str
    args: list[Any]
    kwargs: list[Any]


class SpecStack:
    def __init__(self, env):
        if type(env) == str:
            self.stack = self.deserialise_spec_stack(env)
            self.stack_json = env
        else:
            self.stack = self.spec_stack(env)
            self.stack_json = self.serialise_spec_stack()

    def spec_stack(self, env) -> tuple[Union[WrapperSpec, EnvSpec]]:
        wrapper_spec = WrapperSpec(type(env).__name__, env.__module__ + ":" + type(env).__name__, env._ezpickle_args, env._ezpickle_kwargs)
        if isinstance(env.env, Wrapper):
             return (wrapper_spec,) + self.spec_stack(env.env)
        else:
             return (wrapper_spec,) + (env.env.spec,)


    def serialise_spec_stack(self) -> str:
        num_layers = len(self.stack)
        stack_json = {}
        for i, spec in enumerate(self.stack):
            spec = copy.deepcopy(spec)  # we need to make a copy so we don't modify the original spec in case of callables
            for k, v in spec.kwargs.items():
                if callable(v):
                    str_repr = str(inspect.getsourcelines(v)[0]).strip("['\\n']").split(" = ")[1]  # https://stackoverflow.com/a/30984012
                    str_repr = re.search(r", (.*)\)$", str_repr).group(1)
                    spec.kwargs[k] = str_repr
            if i == num_layers - 1:
                layer = "raw_env"
            else:
                layer = f"wrapper_{num_layers - i - 2}"
            spec_json = json.dumps(dataclasses.asdict(spec))
            stack_json[layer] = spec_json
        return stack_json


    def deserialise_spec_stack(self, stack_json: str, eval_ok: bool = False) -> tuple[Union[WrapperSpec, EnvSpec]]:
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

            for k, v in spec['kwargs'].items():
                if type(v) == str and v[:7] == 'lambda ':
                    if eval_ok:
                        spec['kwargs'][k] = eval(v)
                    else:
                        raise gym.error.Error("Cannot eval lambda functions. Set eval_ok=True to allow this.")

            if name == "raw_env":
                for key in ['namespace', 'name', 'version']:  # remove args where init is set to False
                    spec.pop(key)
                spec = EnvSpec(**spec)
            else:
                spec = WrapperSpec(**spec)
            stack.append(spec)

        return tuple(stack)
        #WrapperSpec(*list(json.loads(stack_json['wrapper_4'].replace("\'", "\"")).values()))

    def __str__(self) -> None:
        table = '\n'
        table += f"{'' :<16} | {' Name' :<26} | {' Parameters' :<50}\n"
        table += "-"*100 + "\n"
        for i in range(len(self.stack)):
            spec = self.stack[-1 - i]
            if type(spec) == WrapperSpec:
                table += f"Wrapper {i-1}:{'' :<6} |  {spec.name :<25} |  {spec.kwargs}\n"
            else:
                table += f"Raw Environment: |  {spec.id :<25} |  {spec.kwargs}\n"
        return table

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
#env = gym.wrappers.TransformReward(env, lambda r: 0.01 * r)
env = gym.wrappers.ResizeObservation(env, (84, 84))

# get the spec stack
stack = SpecStack(env)

# jsonise the spec stack
serialised_stack = stack.stack_json

print(stack)

# deserialise the spec stack
#deserialised_stack = deserialise_spec_stack(serialised_stack, eval_ok=True)
#assert deserialised_stack == stack

# reconstruct the environment
#reconstructed_env = reconstruct_env(deserialised_stack)
#assert spec_stack(reconstructed_env) == spec_stack(env)

print("Done")

# todo: make ezpickle calls into kwargs not args
