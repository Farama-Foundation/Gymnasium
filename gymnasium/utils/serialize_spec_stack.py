import dataclasses
import inspect
import json
import re
from typing import Union

from gymnasium import error
from gymnasium.dataclasses import WrapperSpec
from gymnasium.envs.registration import EnvSpec


def serialise_spec_stack(stack) -> str:
    """Serialises the specification stack into a JSON string.

    Returns:
        A JSON string representing the specification stack.
    """
    num_layers = len(stack)
    stack_json = {}
    for i, spec in enumerate(stack):
        spec = _serialise_callable(spec)
        if i == num_layers - 1:
            layer = "raw_env"
        else:
            layer = f"wrapper_{num_layers - i - 2}"
        spec_json = json.dumps(dataclasses.asdict(spec))
        stack_json[layer] = spec_json
    return stack_json


def _serialise_callable(spec):
    for k, v in spec.kwargs.items():
        if callable(v):
            str_repr = (
                str(inspect.getsourcelines(v)[0])
                .strip("['\\n']")
                .split(" = ")[1]
            )  # https://stackoverflow.com/a/30984012
            str_repr = re.search(r", (.*)\)$", str_repr).group(1)
            spec.kwargs[k] = str_repr
    return spec


def deserialise_spec_stack(
    stack_json: str, eval_ok: bool = False
) -> "tuple[Union[WrapperSpec, EnvSpec]]":
    """Converts a JSON string into a specification stack.

    Args:
        stack_json: The JSON string representing the specification stack.
        eval_ok: Whether to allow evaluation of callables (potentially arbitrary code).

    Returns:
        A tuple of environment and wrapper specifications, known as the specification stack.
    """
    stack = []
    for name, spec_json in stack_json.items():
        spec = json.loads(spec_json)

        # convert lists back into tuples - json saves tuples as lists, so we need to convert them back (assumes depth <2, todo: recursify this)
        for k, v in spec[
            "kwargs"
        ].items():
            if type(v) == list:
                for i, x in enumerate(v):
                    if type(x) == list:
                        spec["kwargs"][k][i] = tuple(x)
                spec["kwargs"][k] = tuple(v)

        # search for lambda functions (as strings) and convert them back to callables
        for k, v in spec["kwargs"].items():
            if type(v) == str and v[:7] == "lambda ":
                if eval_ok:
                    spec["kwargs"][k] = eval(v)
                else:
                    raise error.Error(
                        "Cannot eval lambda functions. Set eval_ok=True to allow this."
                    )

        if name == "raw_env":
            for key in [
                "namespace",
                "name",
                "version",
            ]:  # remove args where init is set to False
                spec.pop(key)
            spec = EnvSpec(**spec)
        else:
            spec = WrapperSpec(**spec)
        stack.append(spec)

    return tuple(stack)


def pprint_stack(spec_json) -> None:
    """Pretty prints the specification stack."""
    table = "\n"
    table += f"{'' :<16} | {' Name' :<26} | {' Parameters' :<50}\n"
    table += "-" * 100 + "\n"
    for layer, spec in reversed(spec_json.items()):
        spec = json.loads(spec)
        if layer == "raw_env":
            table += f"{layer :<16} |  {spec['id'] :<25} |  {spec['kwargs']}\n"
        else:
            table += f"{layer :<16} |  {spec['name'] :<25} |  {spec['kwargs']}\n"
    print(table)