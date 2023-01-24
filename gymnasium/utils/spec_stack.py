"""Functions that use/modify specification stacks."""
from __future__ import annotations

import dataclasses
import json
from typing import Any

from gymnasium.envs.registration import EnvSpec, SpecStack, WrapperSpec
from gymnasium.logger import warn
from gymnasium.spaces import Sequence


def serialize_spec_stack(spec_stack: SpecStack) -> str:
    """Serializes the specification stack into a JSON string.

    Args:
        spec_stack: Tuple of environment and wrapper specifications, known as the specification stack. Generated via `env.spec_stack`.

    Returns:
        A JSON string representing the specification stack.
    """
    assert isinstance(spec_stack, tuple), type(spec_stack)
    assert all(isinstance(spec, WrapperSpec) for spec in spec_stack[:-1])
    assert isinstance(spec_stack[-1], EnvSpec)

    serialized_spec: list[dict[str, Any]] = []
    for spec in spec_stack:
        spec_dict = dataclasses.asdict(spec)
        _check_spec_serialization(spec_dict)

        serialized_spec.append(spec_dict)

    serialized_env_spec = serialized_spec[-1]
    serialized_env_spec.pop("namespace")
    serialized_env_spec.pop("name")
    serialized_env_spec.pop("version")

    return json.dumps(serialized_spec)


def _check_spec_serialization(spec: dict[str, Any]):
    """Warns the user about serialisation failing if the spec contains a callable.

    Args:
        spec: An environment or wrapper specification.

    Returns: The specification with lambda functions converted to strings.

    """
    spec_name = spec["name"] if "name" in spec else spec["id"]

    for key, value in spec.items():
        if callable(value):
            warn(
                f"Callable found in {spec_name} for {key} attribute with value={value}. Currently, Gymnasium does not support serialising callables."
            )


def deserialize_spec_stack(spec_stack: str) -> SpecStack:
    """Converts a JSON string into a specification stack.

    Args:
        spec_stack: The JSON string representing the specification stack.

    Returns:
        A tuple of environment and wrapper specifications, known as the specification stack.
    """
    spec_stack_json = json.loads(spec_stack)

    wrapper_spec_stack = ()
    for wrapper_spec_json in spec_stack_json[:-1]:
        try:
            wrapper_spec_stack += (WrapperSpec(**wrapper_spec_json),)
        except Exception as e:
            raise ValueError(
                f"An issue occurred when trying to make {wrapper_spec_json} a WrapperSpec"
            ) from e

    env_spec_json = spec_stack_json[-1]
    try:
        env_spec = EnvSpec(**env_spec_json)
    except Exception as e:
        raise ValueError(
            f"An issue occurred when trying to make {env_spec_json} an EnvSpec"
        ) from e

    return wrapper_spec_stack + (env_spec,)


def pprint_spec_stack(
    spec_stack: SpecStack | str | list[dict[str, Any]],
    disable_print: bool = False,
) -> str | None:
    """Pretty prints the specification stack either from ``env.spec_stack`` or JSON string from ``serialize_spec_stack``.

    Args:
        spec_stack: The spec stack either from `env.spec_stack` or the JSON string / dict representing the specification stack generated via ``serialize_spec_stack(env.spec_stack)``.
        disable_print: If to disable printing the spec stack, by default false allowing the spec to be printed.
    """
    if isinstance(spec_stack, tuple):
        assert all(isinstance(spec, WrapperSpec) for spec in spec_stack[:-1])
        assert isinstance(spec_stack[-1], EnvSpec)

        table = "\n"
        table += f"{'' :<16} | {' Name' :<26} | {' Parameters' :<50}\n"
        table += "-" * 100 + "\n"

        for spec in spec_stack[:-1]:
            table += f"{spec.name :<25} |  {spec.kwargs}\n"
        table += f"{spec_stack[-1].id :<25} |  {spec_stack[-1].kwargs}\n"

    elif isinstance(spec_stack, str) or isinstance(spec_stack, list):
        if isinstance(spec_stack, str):
            spec_stack_json = json.loads(spec_stack)
        else:
            assert all(isinstance(spec, dict) for spec in spec_stack)
            spec_stack_json = spec_stack

        table = "\n"
        table += f"{' Name' :<26} | {' Parameters' :<50}\n"
        table += "-" * 100 + "\n"

        for spec_json in spec_stack_json[:-1]:
            table += f"{spec_json['name'] :<25} |  {spec_json['kwargs']}\n"
        table += (
            f"{spec_stack_json[-1]['id'] :<25} |  {spec_stack_json[-1]['kwargs']}\n"
        )
    else:
        raise TypeError(
            f"The `spec_stack` to expected to be tuple or list type, actual type: {type(spec_stack)}"
        )

    if disable_print:
        return table
    else:
        print(table, end="")
