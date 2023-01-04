"""Functions that use/modify specification stacks."""
import dataclasses
import inspect
import json
import re
from typing import Union

from gymnasium import error
from gymnasium.envs.registration import EnvSpec
from gymnasium.logger import warn
from gymnasium.wrapperspec import WrapperSpec


def serialise_spec_stack(stack: "tuple[Union[WrapperSpec, EnvSpec]]") -> str:
    """Serialises the specification stack into a JSON string.

    Args:
        stack: Tuple of environment and wrapper specifications, known as the specification stack. Generated via `env.spec_stack`.

    Returns:
        A JSON string representing the specification stack.
    """
    num_layers = len(stack)
    stack_json = {}
    for i, spec in enumerate(stack):
        spec = _warn_callable(spec)
        if i == num_layers - 1:
            if isinstance(spec, WrapperSpec) or not isinstance(
                spec, EnvSpec
            ):  # NB: WrapperSpec is a subclass of EnvSpec
                raise error.Error(
                    f"Expected spec to be an EnvSpec, but got {type(spec)} instead."
                )
            layer = "raw_env"
        else:
            if not isinstance(spec, WrapperSpec):
                raise error.Error(
                    f"Expected spec to be a WrapperSpec, but got {type(spec)} instead."
                )
            layer = f"wrapper_{num_layers - i - 2}"
        spec_json = json.dumps(dataclasses.asdict(spec))
        stack_json[layer] = spec_json
    return stack_json


def _warn_callable(spec: Union[WrapperSpec, EnvSpec]) -> dict:
    """Warns the user about serialisation failing if the spec contains a callable.

    Args:
        spec: An environment or wrapper specification.

    Returns: The specification with lambda functions converted to strings.

    """
    for k, v in spec.kwargs.items():
        if callable(v):
            warn("Callable found in environment specification stack. This likely comes from a wrapper that inputs a function. "
                 "Currently, Gymnasium does not support serialising callables. This will be fixed in a future release.")
    return spec


def deserialise_spec_stack(
    stack_json: dict, eval_ok: bool = False
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
        for k, v in spec["kwargs"].items():
            if type(v) == list:
                for i, x in enumerate(v):
                    if type(x) == list:
                        spec["kwargs"][k][i] = tuple(x)
                spec["kwargs"][k] = tuple(v)

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


def pprint_stack(spec_json: dict) -> None:
    """Pretty prints the specification stack.

    Args:
        spec_json: The JSON string representing the specification stack. Generated via `serialise_spec_stack(env.spec_stack)`.
    """
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
