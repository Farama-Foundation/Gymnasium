"""Functions for registering environments within gymnasium using public functions ``make``, ``register`` and ``spec``."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import difflib
import importlib
import importlib.util
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence

import gymnasium as gym
from gymnasium import Env, Wrapper, error, logger
from gymnasium.logger import warn
from gymnasium.vector import AutoresetMode


if sys.version_info < (3, 10):
    import importlib_metadata as metadata  # type: ignore
else:
    import importlib.metadata as metadata

from typing import Protocol


ENV_ID_RE = re.compile(
    r"^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)


__all__ = [
    "registry",
    "current_namespace",
    "EnvSpec",
    "WrapperSpec",
    "VectorizeMode",
    # Functions
    "register",
    "make",
    "make_vec",
    "spec",
    "pprint_registry",
    "register_envs",
]


class EnvCreator(Protocol):
    """Function type expected for an environment."""

    def __call__(self, **kwargs: Any) -> Env: ...


class VectorEnvCreator(Protocol):
    """Function type expected for an environment."""

    def __call__(self, **kwargs: Any) -> gym.vector.VectorEnv: ...


@dataclass
class WrapperSpec:
    """A specification for recording wrapper configs.

    * name: The name of the wrapper.
    * entry_point: The location of the wrapper to create from.
    * kwargs: Additional keyword arguments passed to the wrapper. If the wrapper doesn't inherit from EzPickle then this is ``None``
    """

    name: str
    entry_point: str
    kwargs: dict[str, Any] | None


@dataclass
class EnvSpec:
    """A specification for creating environments with :meth:`gymnasium.make`.

    * **id**: The string used to create the environment with :meth:`gymnasium.make`
    * **entry_point**: A string for the environment location, ``(import path):(environment name)`` or a function that creates the environment.
    * **reward_threshold**: The reward threshold for completing the environment.
    * **nondeterministic**: If the observation of an environment cannot be repeated with the same initial state, random number generator state and actions.
    * **max_episode_steps**: The max number of steps that the environment can take before truncation
    * **order_enforce**: If to enforce the order of :meth:`gymnasium.Env.reset` before :meth:`gymnasium.Env.step` and :meth:`gymnasium.Env.render` functions
    * **disable_env_checker**: If to disable the environment checker wrapper in :meth:`gymnasium.make`, by default False (runs the environment checker)
    * **kwargs**: Additional keyword arguments passed to the environment during initialisation
    * **additional_wrappers**: A tuple of additional wrappers applied to the environment (WrapperSpec)
    * **vector_entry_point**: The location of the vectorized environment to create from

    Changelogs:
        v1.0.0 - Autoreset attribute removed
    """

    id: str
    entry_point: EnvCreator | str | None = field(default=None)

    # Environment attributes
    reward_threshold: float | None = field(default=None)
    nondeterministic: bool = field(default=False)

    # Wrappers
    max_episode_steps: int | None = field(default=None)
    order_enforce: bool = field(default=True)
    disable_env_checker: bool = field(default=False)

    # Environment arguments
    kwargs: dict = field(default_factory=dict)

    # post-init attributes
    namespace: str | None = field(init=False)
    name: str = field(init=False)
    version: int | None = field(init=False)

    # applied wrappers
    additional_wrappers: tuple[WrapperSpec, ...] = field(default_factory=tuple)

    # Vectorized environment entry point
    vector_entry_point: VectorEnvCreator | str | None = field(default=None)

    def __post_init__(self):
        """Calls after the spec is created to extract the namespace, name and version from the environment id."""
        self.namespace, self.name, self.version = parse_env_id(self.id)

    def make(self, **kwargs: Any) -> Env:
        """Calls ``make`` using the environment spec and any keyword arguments."""
        return make(self, **kwargs)

    def to_json(self) -> str:
        """Converts the environment spec into a json compatible string.

        Returns:
            A jsonifyied string for the environment spec
        """
        env_spec_dict = dataclasses.asdict(self)
        # As the namespace, name and version are initialised after `init` then we remove the attributes
        env_spec_dict.pop("namespace")
        env_spec_dict.pop("name")
        env_spec_dict.pop("version")

        # To check that the environment spec can be transformed to a json compatible type
        self._check_can_jsonify(env_spec_dict)

        return json.dumps(env_spec_dict)

    @staticmethod
    def _check_can_jsonify(env_spec: dict[str, Any]):
        """Warns the user about serialisation failing if the spec contains a callable.

        Args:
            env_spec: An environment or wrapper specification.

        Returns: The specification with lambda functions converted to strings.

        """
        spec_name = env_spec["name"] if "name" in env_spec else env_spec["id"]

        for key, value in env_spec.items():
            if callable(value):
                raise ValueError(
                    f"Callable found in {spec_name} for {key} attribute with value={value}. Currently, Gymnasium does not support serialising callables."
                )

    @staticmethod
    def from_json(json_env_spec: str) -> EnvSpec:
        """Converts a JSON string into a specification stack.

        Args:
            json_env_spec: A JSON string representing the env specification.

        Returns:
            An environment spec
        """
        parsed_env_spec = json.loads(json_env_spec)

        applied_wrapper_specs: list[WrapperSpec] = []
        for wrapper_spec_json in parsed_env_spec.pop("additional_wrappers"):
            try:
                applied_wrapper_specs.append(WrapperSpec(**wrapper_spec_json))
            except Exception as e:
                raise ValueError(
                    f"An issue occurred when trying to make {wrapper_spec_json} a WrapperSpec"
                ) from e

        try:
            env_spec = EnvSpec(**parsed_env_spec)
            env_spec.additional_wrappers = tuple(applied_wrapper_specs)
        except Exception as e:
            raise ValueError(
                f"An issue occurred when trying to make {parsed_env_spec} an EnvSpec"
            ) from e

        return env_spec

    def pprint(
        self,
        disable_print: bool = False,
        include_entry_points: bool = False,
        print_all: bool = False,
    ) -> str | None:
        """Pretty prints the environment spec.

        Args:
            disable_print: If to disable print and return the output
            include_entry_points: If to include the entry_points in the output
            print_all: If to print all information, including variables with default values

        Returns:
            If ``disable_print is True`` a string otherwise ``None``
        """
        output = f"id={self.id}"
        if print_all or include_entry_points:
            output += f"\nentry_point={self.entry_point}"

        if print_all or self.reward_threshold is not None:
            output += f"\nreward_threshold={self.reward_threshold}"
        if print_all or self.nondeterministic is not False:
            output += f"\nnondeterministic={self.nondeterministic}"

        if print_all or self.max_episode_steps is not None:
            output += f"\nmax_episode_steps={self.max_episode_steps}"
        if print_all or self.order_enforce is not True:
            output += f"\norder_enforce={self.order_enforce}"
        if print_all or self.disable_env_checker is not False:
            output += f"\ndisable_env_checker={self.disable_env_checker}"

        if print_all or self.additional_wrappers:
            wrapper_output: list[str] = []
            for wrapper_spec in self.additional_wrappers:
                if include_entry_points:
                    wrapper_output.append(
                        f"\n\tname={wrapper_spec.name}, entry_point={wrapper_spec.entry_point}, kwargs={wrapper_spec.kwargs}"
                    )
                else:
                    wrapper_output.append(
                        f"\n\tname={wrapper_spec.name}, kwargs={wrapper_spec.kwargs}"
                    )

            if len(wrapper_output) == 0:
                output += "\nadditional_wrappers=[]"
            else:
                output += f"\nadditional_wrappers=[{','.join(wrapper_output)}\n]"

        if disable_print:
            return output
        else:
            print(output)


class VectorizeMode(Enum):
    """All possible vectorization modes used in `make_vec`."""

    ASYNC = "async"
    SYNC = "sync"
    VECTOR_ENTRY_POINT = "vector_entry_point"


# Global registry of environments. Meant to be accessed through `register` and `make`
registry: dict[str, EnvSpec] = {}
current_namespace: str | None = None


def parse_env_id(env_id: str) -> tuple[str | None, str, int | None]:
    """Parse environment ID string format - ``[namespace/](env-name)[-v(version)]`` where the namespace and version are optional.

    Args:
        env_id: The environment id to parse

    Returns:
        A tuple of environment namespace, environment name and version number

    Raises:
        Error: If the environment id is not valid environment regex
    """
    match = ENV_ID_RE.fullmatch(env_id)
    if not match:
        raise error.Error(
            f"Malformed environment ID: {env_id}. (Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))"
        )
    ns, name, version = match.group("namespace", "name", "version")
    if version is not None:
        version = int(version)

    return ns, name, version


def get_env_id(ns: str | None, name: str, version: int | None) -> str:
    """Get the full env ID given a name and (optional) version and namespace. Inverse of :meth:`parse_env_id`.

    Args:
        ns: The environment namespace
        name: The environment name
        version: The environment version

    Returns:
        The environment id
    """
    full_name = name
    if ns is not None:
        full_name = f"{ns}/{name}"
    if version is not None:
        full_name = f"{full_name}-v{version}"

    return full_name


def find_highest_version(ns: str | None, name: str) -> int | None:
    """Finds the highest registered version of the environment given the namespace and name in the registry.

    Args:
        ns: The environment namespace
        name: The environment name (id)

    Returns:
        The highest version of an environment with matching namespace and name, otherwise ``None`` is returned.
    """
    version: list[int] = [
        env_spec.version
        for env_spec in registry.values()
        if env_spec.namespace == ns
        and env_spec.name == name
        and env_spec.version is not None
    ]
    return max(version, default=None)


def _check_namespace_exists(ns: str | None):
    """Check if a namespace exists. If it doesn't, print a helpful error message."""
    # If the namespace is none, then the namespace does exist
    if ns is None:
        return

    # Check if the namespace exists in one of the registry's specs
    namespaces: set[str] = {
        env_spec.namespace
        for env_spec in registry.values()
        if env_spec.namespace is not None
    }
    if ns in namespaces:
        return

    # Otherwise, the namespace doesn't exist and raise a helpful message
    suggestion = (
        difflib.get_close_matches(ns, namespaces, n=1) if len(namespaces) > 0 else None
    )
    if suggestion:
        suggestion_msg = f"Did you mean: `{suggestion[0]}`?"
    else:
        suggestion_msg = f"Have you installed the proper package for {ns}?"

    raise error.NamespaceNotFound(f"Namespace {ns} not found. {suggestion_msg}")


def _check_name_exists(ns: str | None, name: str):
    """Check if an env exists in a namespace. If it doesn't, print a helpful error message."""
    # First check if the namespace exists
    _check_namespace_exists(ns)

    # Then check if the name exists
    names: set[str] = {
        env_spec.name for env_spec in registry.values() if env_spec.namespace == ns
    }
    if name in names:
        return

    # Otherwise, raise a helpful error to the user
    suggestion = difflib.get_close_matches(name, names, n=1)
    namespace_msg = f" in namespace {ns}" if ns else ""
    suggestion_msg = f" Did you mean: `{suggestion[0]}`?" if suggestion else ""

    raise error.NameNotFound(
        f"Environment `{name}` doesn't exist{namespace_msg}.{suggestion_msg}"
    )


def _check_version_exists(ns: str | None, name: str, version: int | None):
    """Check if an env version exists in a namespace. If it doesn't, print a helpful error message.

    This is a complete test whether an environment identifier is valid, and will provide the best available hints.

    Args:
        ns: The environment namespace
        name: The environment space
        version: The environment version

    Raises:
        DeprecatedEnv: The environment doesn't exist but a default version does
        VersionNotFound: The ``version`` used doesn't exist
        DeprecatedEnv: Environment version is deprecated
    """
    if get_env_id(ns, name, version) in registry:
        return

    _check_name_exists(ns, name)
    if version is None:
        return

    message = f"Environment version `v{version}` for environment `{get_env_id(ns, name, None)}` doesn't exist."

    env_specs = [
        env_spec
        for env_spec in registry.values()
        if env_spec.namespace == ns and env_spec.name == name
    ]
    env_specs = sorted(env_specs, key=lambda env_spec: int(env_spec.version or -1))

    default_spec = [env_spec for env_spec in env_specs if env_spec.version is None]

    if default_spec:
        message += f" It provides the default version `{default_spec[0].id}`."
        if len(env_specs) == 1:
            raise error.DeprecatedEnv(message)

    # Process possible versioned environments

    versioned_specs = [
        env_spec for env_spec in env_specs if env_spec.version is not None
    ]

    latest_spec = max(versioned_specs, key=lambda env_spec: env_spec.version, default=None)  # type: ignore
    if latest_spec is not None and version > latest_spec.version:
        version_list_msg = ", ".join(f"`v{env_spec.version}`" for env_spec in env_specs)
        message += f" It provides versioned environments: [ {version_list_msg} ]."

        raise error.VersionNotFound(message)

    if latest_spec is not None and version < latest_spec.version:
        raise error.DeprecatedEnv(
            f"Environment version v{version} for `{get_env_id(ns, name, None)}` is deprecated. "
            f"Please use `{latest_spec.id}` instead."
        )


def _check_spec_register(testing_spec: EnvSpec):
    """Checks whether the spec is valid to be registered. Helper function for `register`."""
    latest_versioned_spec = max(
        (
            env_spec
            for env_spec in registry.values()
            if env_spec.namespace == testing_spec.namespace
            and env_spec.name == testing_spec.name
            and env_spec.version is not None
        ),
        key=lambda spec_: int(spec_.version),  # type: ignore
        default=None,
    )

    unversioned_spec = next(
        (
            env_spec
            for env_spec in registry.values()
            if env_spec.namespace == testing_spec.namespace
            and env_spec.name == testing_spec.name
            and env_spec.version is None
        ),
        None,
    )

    if unversioned_spec is not None and testing_spec.version is not None:
        raise error.RegistrationError(
            "Can't register the versioned environment "
            f"`{testing_spec.id}` when the unversioned environment "
            f"`{unversioned_spec.id}` of the same name already exists."
        )
    elif latest_versioned_spec is not None and testing_spec.version is None:
        raise error.RegistrationError(
            f"Can't register the unversioned environment `{testing_spec.id}` when the versioned environment "
            f"`{latest_versioned_spec.id}` of the same name already exists. Note: the default behavior is "
            "that `gym.make` with the unversioned environment will return the latest versioned environment"
        )


def _check_metadata(testing_metadata: dict[str, Any]):
    """Check the metadata of an environment."""
    if not isinstance(testing_metadata, dict):
        raise error.InvalidMetadata(
            f"Expect the environment metadata to be dict, actual type: {type(metadata)}"
        )

    render_modes = testing_metadata.get("render_modes")
    if render_modes is None:
        logger.warn(
            f"The environment creator metadata doesn't include `render_modes`, contains: {list(testing_metadata.keys())}"
        )
    elif not isinstance(render_modes, Iterable):
        logger.warn(
            f"Expects the environment metadata render_modes to be a Iterable, actual type: {type(render_modes)}"
        )


def _find_spec(env_id: str) -> EnvSpec:
    # For string id's, load the environment spec from the registry then make the environment spec
    assert isinstance(env_id, str)

    # The environment name can include an unloaded module in "module:env_name" style
    module, env_name = (None, env_id) if ":" not in env_id else env_id.split(":")
    if module is not None:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}. Environment registration via importing a module failed. "
                f"Check whether '{module}' contains env registration and can be imported."
            ) from e

    # load the env spec from the registry
    env_spec = registry.get(env_name)

    # update env spec is not version provided, raise warning if out of date
    ns, name, version = parse_env_id(env_name)

    latest_version = find_highest_version(ns, name)
    if version is not None and latest_version is not None and latest_version > version:
        logger.deprecation(
            f"The environment {env_name} is out of date. You should consider "
            f"upgrading to version `v{latest_version}`."
        )
    if version is None and latest_version is not None:
        version = latest_version
        new_env_id = get_env_id(ns, name, version)
        env_spec = registry.get(new_env_id)
        logger.warn(
            f"Using the latest versioned environment `{new_env_id}` "
            f"instead of the unversioned environment `{env_name}`."
        )

    if env_spec is None:
        _check_version_exists(ns, name, version)
        raise error.Error(
            f"No registered env with id: {env_name}. Did you register it, or import the package that registers it? Use `gymnasium.pprint_registry()` to see all of the registered environments."
        )

    return env_spec


def load_env_creator(name: str) -> EnvCreator | VectorEnvCreator:
    """Loads an environment with name of style ``"(import path):(environment name)"`` and returns the environment creation function, normally the environment class type.

    Args:
        name: The environment name

    Returns:
        The environment constructor for the given environment name.
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def register_envs(env_module: ModuleType):
    """A No-op function such that it can appear to IDEs that a module is used."""
    pass


@contextlib.contextmanager
def namespace(ns: str):
    """Context manager for modifying the current namespace."""
    global current_namespace
    old_namespace = current_namespace
    current_namespace = ns
    yield
    current_namespace = old_namespace


def register(
    id: str,
    entry_point: EnvCreator | str | None = None,
    reward_threshold: float | None = None,
    nondeterministic: bool = False,
    max_episode_steps: int | None = None,
    order_enforce: bool = True,
    disable_env_checker: bool = False,
    additional_wrappers: tuple[WrapperSpec, ...] = (),
    vector_entry_point: VectorEnvCreator | str | None = None,
    kwargs: dict | None = None,
):
    """Registers an environment in gymnasium with an ``id`` to use with :meth:`gymnasium.make` with the ``entry_point`` being a string or callable for creating the environment.

    The ``id`` parameter corresponds to the name of the environment, with the syntax as follows:
    ``[namespace/](env_name)[-v(version)]`` where ``namespace`` and ``-v(version)`` is optional.

    It takes arbitrary keyword arguments, which are passed to the :class:`EnvSpec` ``kwargs`` parameter.

    Args:
        id: The environment id
        entry_point: The entry point for creating the environment
        reward_threshold: The reward threshold considered for an agent to have learnt the environment
        nondeterministic: If the environment is nondeterministic (even with knowledge of the initial seed and all actions, the same state cannot be reached)
        max_episode_steps: The maximum number of episodes steps before truncation. Used by the :class:`gymnasium.wrappers.TimeLimit` wrapper if not ``None``.
        order_enforce: If to enable the order enforcer wrapper to ensure users run functions in the correct order.
            If ``True``, then the :class:`gymnasium.wrappers.OrderEnforcing` is applied to the environment.
        disable_env_checker: If to disable the :class:`gymnasium.wrappers.PassiveEnvChecker` to the environment.
        additional_wrappers: Additional wrappers to apply the environment.
        vector_entry_point: The entry point for creating the vector environment
        kwargs: arbitrary keyword arguments which are passed to the environment constructor on initialisation.

    Changelogs:
        v1.0.0 - `autoreset` and `apply_api_compatibility` parameter was removed
    """
    assert (
        entry_point is not None or vector_entry_point is not None
    ), "Either `entry_point` or `vector_entry_point` (or both) must be provided"
    global registry, current_namespace
    ns, name, version = parse_env_id(id)

    if kwargs is None:
        kwargs = dict()
    if current_namespace is not None:
        if (
            kwargs.get("namespace") is not None
            and kwargs.get("namespace") != current_namespace
        ):
            logger.warn(
                f"Custom namespace `{kwargs.get('namespace')}` is being overridden by namespace `{current_namespace}`. "
                f"If you are developing a plugin you shouldn't specify a namespace in `register` calls. "
                "The namespace is specified through the entry point package metadata."
            )
        ns_id = current_namespace
    else:
        ns_id = ns
    full_env_id = get_env_id(ns_id, name, version)

    new_spec = EnvSpec(
        id=full_env_id,
        entry_point=entry_point,
        reward_threshold=reward_threshold,
        nondeterministic=nondeterministic,
        max_episode_steps=max_episode_steps,
        order_enforce=order_enforce,
        disable_env_checker=disable_env_checker,
        kwargs=kwargs,
        additional_wrappers=additional_wrappers,
        vector_entry_point=vector_entry_point,
    )
    _check_spec_register(new_spec)

    if new_spec.id in registry:
        logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    registry[new_spec.id] = new_spec


def make(
    id: str | EnvSpec,
    max_episode_steps: int | None = None,
    disable_env_checker: bool | None = None,
    **kwargs: Any,
) -> Env:
    """Creates an environment previously registered with :meth:`gymnasium.register` or a :class:`EnvSpec`.

    To find all available environments use ``gymnasium.envs.registry.keys()`` for all valid ids.

    Args:
        id: A string for the environment id or a :class:`EnvSpec`. Optionally if using a string, a module to import can be included, e.g. ``'module:Env-v0'``.
            This is equivalent to importing the module first to register the environment followed by making the environment.
        max_episode_steps: Maximum length of an episode, can override the registered :class:`EnvSpec` ``max_episode_steps``
            with the value being passed to :class:`gymnasium.wrappers.TimeLimit`.
            Using ``max_episode_steps=-1`` will not apply the wrapper to the environment.
        disable_env_checker: If to add :class:`gymnasium.wrappers.PassiveEnvChecker`, ``None`` will default to the
            :class:`EnvSpec` ``disable_env_checker`` value otherwise use this value will be used.
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment with wrappers applied.

    Raises:
        Error: If the ``id`` doesn't exist in the :attr:`registry`

    Changelogs:
        v1.0.0 - `autoreset` and `apply_api_compatibility` was removed
    """
    if isinstance(id, EnvSpec):
        env_spec = id
        if not hasattr(env_spec, "additional_wrappers"):
            logger.warn(
                f"The env spec passed to `make` does not have a `additional_wrappers`, set it to an empty tuple. Env_spec={env_spec}"
            )
            env_spec.additional_wrappers = ()
    else:
        # For string id's, load the environment spec from the registry then make the environment spec
        assert isinstance(id, str)

        # The environment name can include an unloaded module in "module:env_name" style
        env_spec = _find_spec(id)

    assert isinstance(env_spec, EnvSpec)

    # Update the env spec kwargs with the `make` kwargs
    env_spec_kwargs = copy.deepcopy(env_spec.kwargs)
    env_spec_kwargs.update(kwargs)

    # Load the environment creator
    if env_spec.entry_point is None:
        raise error.Error(f"{env_spec.id} registered but entry_point is not specified")
    elif callable(env_spec.entry_point):
        env_creator = env_spec.entry_point
    else:
        # Assume it's a string
        env_creator = load_env_creator(env_spec.entry_point)

    # Determine if to use the rendering
    render_modes: list[str] | None = None
    if hasattr(env_creator, "metadata"):
        _check_metadata(env_creator.metadata)
        render_modes = env_creator.metadata.get("render_modes")
    render_mode = env_spec_kwargs.get("render_mode")
    apply_human_rendering = False
    apply_render_collection = False

    # If mode is not valid, try applying HumanRendering/RenderCollection wrappers
    if (
        render_mode is not None
        and render_modes is not None
        and render_mode not in render_modes
    ):
        displayable_modes = {"rgb_array", "rgb_array_list"}.intersection(render_modes)
        if render_mode == "human" and len(displayable_modes) > 0:
            logger.warn(
                "You are trying to use 'human' rendering for an environment that doesn't natively support it. "
                "The HumanRendering wrapper is being applied to your environment."
            )
            env_spec_kwargs["render_mode"] = displayable_modes.pop()
            apply_human_rendering = True
        elif (
            render_mode.endswith("_list")
            and render_mode[: -len("_list")] in render_modes
        ):
            env_spec_kwargs["render_mode"] = render_mode[: -len("_list")]
            apply_render_collection = True
        else:
            logger.warn(
                f"The environment is being initialised with render_mode={render_mode!r} "
                f"that is not in the possible render_modes ({render_modes})."
            )

    try:
        env = env_creator(**env_spec_kwargs)
    except TypeError as e:
        if (
            str(e).find("got an unexpected keyword argument 'render_mode'") >= 0
            and apply_human_rendering
        ):
            raise error.Error(
                f"You passed render_mode='human' although {env_spec.id} doesn't implement human-rendering natively. "
                "Gym tried to apply the HumanRendering wrapper but it looks like your environment is using the old "
                "rendering API, which is not supported by the HumanRendering wrapper."
            ) from e
        else:
            raise type(e)(
                f"{e} was raised from the environment creator for {env_spec.id} with kwargs ({env_spec_kwargs})"
            )

    if not isinstance(env, gym.Env):
        if (
            str(env.__class__.__base__) == "<class 'gym.core.Env'>"
            or str(env.__class__.__base__) == "<class 'gym.core.Wrapper'>"
        ):
            raise TypeError(
                "Gym is incompatible with Gymnasium, please update the environment class to `gymnasium.Env`. "
                "See https://gymnasium.farama.org/introduction/create_custom_env/ for more info."
            )
        else:
            raise TypeError(
                f"The environment must inherit from the gymnasium.Env class, actual class: {type(env)}. "
                "See https://gymnasium.farama.org/introduction/create_custom_env/ for more info."
            )

    # Set the minimal env spec for the environment.
    env.unwrapped.spec = EnvSpec(
        id=env_spec.id,
        entry_point=env_spec.entry_point,
        reward_threshold=env_spec.reward_threshold,
        nondeterministic=env_spec.nondeterministic,
        max_episode_steps=None,
        order_enforce=False,
        disable_env_checker=True,
        kwargs=env_spec_kwargs,
        additional_wrappers=(),
        vector_entry_point=env_spec.vector_entry_point,
    )

    # Check if pre-wrapped wrappers
    assert env.spec is not None
    num_prior_wrappers = len(env.spec.additional_wrappers)
    if (
        env_spec.additional_wrappers[:num_prior_wrappers]
        != env.spec.additional_wrappers
    ):
        for env_spec_wrapper_spec, recreated_wrapper_spec in zip(
            env_spec.additional_wrappers, env.spec.additional_wrappers
        ):
            raise ValueError(
                f"The environment's wrapper spec {recreated_wrapper_spec} is different from the saved `EnvSpec` additional wrapper {env_spec_wrapper_spec}"
            )

    # Run the environment checker as the lowest level wrapper
    if disable_env_checker is False or (
        disable_env_checker is None and env_spec.disable_env_checker is False
    ):
        env = gym.wrappers.PassiveEnvChecker(env)

    # Add the order enforcing wrapper
    if env_spec.order_enforce:
        env = gym.wrappers.OrderEnforcing(env)

    # Add the time limit wrapper
    if max_episode_steps != -1:
        if max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps)
        elif env_spec.max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, env_spec.max_episode_steps)

    for wrapper_spec in env_spec.additional_wrappers[num_prior_wrappers:]:
        if wrapper_spec.kwargs is None:
            raise ValueError(
                f"{wrapper_spec.name} wrapper does not inherit from `gymnasium.utils.RecordConstructorArgs`, therefore, the wrapper cannot be recreated."
            )

        env = load_env_creator(wrapper_spec.entry_point)(env=env, **wrapper_spec.kwargs)

    # Add human rendering wrapper
    if apply_human_rendering:
        env = gym.wrappers.HumanRendering(env)
    elif apply_render_collection:
        env = gym.wrappers.RenderCollection(env)

    return env


def make_vec(
    id: str | EnvSpec,
    num_envs: int = 1,
    vectorization_mode: VectorizeMode | str | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    wrappers: Sequence[Callable[[Env], Wrapper]] | None = None,
    **kwargs,
) -> gym.vector.VectorEnv:
    """Create a vector environment according to the given ID.

    To find all available environments use :func:`gymnasium.pprint_registry` or ``gymnasium.registry.keys()`` for all valid ids.
    We refer to the Vector environment as the vectorizor while the environment being vectorized is the base or vectorized environment (``vectorizor(vectorized env)``).

    Args:
        id: Name of the environment. Optionally, a module to import can be included, e.g. 'module:Env-v0'
        num_envs: Number of environments to create
        vectorization_mode: The vectorization method used, defaults to ``None`` such that if env id' spec has a ``vector_entry_point`` (not ``None``),
            this is first used otherwise defaults to ``sync`` to use the :class:`gymnasium.vector.SyncVectorEnv`.
            Valid modes are ``"async"``, ``"sync"`` or ``"vector_entry_point"``. Recommended to use the :class:`VectorizeMode` enum rather than strings.
        vector_kwargs: Additional arguments to pass to the vectorizor environment constructor, i.e., ``SyncVectorEnv(..., **vector_kwargs)``.
        wrappers: A sequence of wrapper functions to apply to the base environment. Can only be used in ``"sync"`` or ``"async"`` mode.
        **kwargs: Additional arguments passed to the base environment constructor.

    Returns:
        An instance of the environment.

    Raises:
        Error: If the ``id`` doesn't exist then an error is raised
    """
    if vector_kwargs is None:
        vector_kwargs = {}
    if wrappers is None:
        wrappers = []

    if isinstance(id, EnvSpec):
        env_spec = id
    elif isinstance(id, str):
        env_spec = _find_spec(id)
    else:
        raise error.Error(f"Invalid id type: {type(id)}. Expected `str` or `EnvSpec`")

    env_spec = copy.deepcopy(env_spec)
    env_spec_kwargs = env_spec.kwargs
    # for sync or async, these parameters should be passed in `make(..., **kwargs)` rather than in the env spec kwargs, therefore, we `reset` the kwargs
    env_spec.kwargs = dict()

    num_envs = env_spec_kwargs.pop("num_envs", num_envs)
    vectorization_mode = env_spec_kwargs.pop("vectorization_mode", vectorization_mode)
    vector_kwargs = env_spec_kwargs.pop("vector_kwargs", vector_kwargs)
    wrappers = env_spec_kwargs.pop("wrappers", wrappers)

    env_spec_kwargs.update(kwargs)

    # Specify the vectorization mode if None or update to a `VectorizeMode`
    if vectorization_mode is None:
        if env_spec.vector_entry_point is not None:
            vectorization_mode = VectorizeMode.VECTOR_ENTRY_POINT
        else:
            vectorization_mode = VectorizeMode.SYNC
    else:
        try:
            vectorization_mode = VectorizeMode(vectorization_mode)
        except ValueError:
            raise ValueError(
                f"Invalid vectorization mode: {vectorization_mode!r}, "
                f"valid modes: {[mode.value for mode in VectorizeMode]}"
            )
    assert isinstance(vectorization_mode, VectorizeMode)

    def create_single_env() -> Env:
        single_env = make(env_spec, **env_spec_kwargs.copy())

        if wrappers is None:
            return single_env

        for wrapper in wrappers:
            single_env = wrapper(single_env)
        return single_env

    if vectorization_mode == VectorizeMode.SYNC:
        if env_spec.entry_point is None:
            raise error.Error(
                f"Cannot create vectorized environment for {env_spec.id} because it doesn't have an entry point defined."
            )

        env = gym.vector.SyncVectorEnv(
            env_fns=(create_single_env for _ in range(num_envs)),
            **vector_kwargs,
        )
    elif vectorization_mode == VectorizeMode.ASYNC:
        if env_spec.entry_point is None:
            raise error.Error(
                f"Cannot create vectorized environment for {env_spec.id} because it doesn't have an entry point defined."
            )

        env = gym.vector.AsyncVectorEnv(
            env_fns=[create_single_env for _ in range(num_envs)],
            **vector_kwargs,
        )

    elif vectorization_mode == VectorizeMode.VECTOR_ENTRY_POINT:
        if len(vector_kwargs) > 0:
            raise error.Error(
                f"Custom vector environment can be passed arguments only through kwargs and `vector_kwargs` is not empty ({vector_kwargs})"
            )
        elif len(wrappers) > 0:
            raise error.Error(
                f"Cannot use `vector_entry_point` vectorization mode with the wrappers argument ({wrappers})."
            )
        elif len(env_spec.additional_wrappers) > 0:
            raise error.Error(
                f"Cannot use `vector_entry_point` vectorization mode with the additional_wrappers parameter in spec being not empty ({env_spec.additional_wrappers})."
            )

        entry_point = env_spec.vector_entry_point
        if entry_point is None:
            raise error.Error(
                f"Cannot create vectorized environment for {id} because it doesn't have a vector entry point defined."
            )
        elif callable(entry_point):
            env_creator = entry_point
        else:  # Assume it's a string
            env_creator = load_env_creator(entry_point)

        if (
            env_spec.max_episode_steps is not None
            and "max_episode_steps" not in env_spec_kwargs
        ):
            env_spec_kwargs["max_episode_steps"] = env_spec.max_episode_steps

        env = env_creator(num_envs=num_envs, **env_spec_kwargs)
    else:
        raise error.Error(f"Unknown vectorization mode: {vectorization_mode}")

    # Copies the environment creation specification and kwargs to add to the environment specification details
    copied_id_spec = copy.deepcopy(env_spec)
    copied_id_spec.kwargs = env_spec_kwargs.copy()
    if num_envs != 1:
        copied_id_spec.kwargs["num_envs"] = num_envs
    copied_id_spec.kwargs["vectorization_mode"] = vectorization_mode.value
    if len(vector_kwargs) > 0:
        copied_id_spec.kwargs["vector_kwargs"] = vector_kwargs
    if len(wrappers) > 0:
        copied_id_spec.kwargs["wrappers"] = wrappers
    env.unwrapped.spec = copied_id_spec

    if "autoreset_mode" not in env.metadata:
        warn(
            f"The VectorEnv ({env}) is missing AutoresetMode metadata, metadata={env.metadata}"
        )
    elif not isinstance(env.metadata["autoreset_mode"], AutoresetMode):
        warn(
            f"The VectorEnv ({env}) metadata['autoreset_mode'] is not an instance of AutoresetMode, {type(env.metadata['autoreset_mode'])}."
        )

    return env


def spec(env_id: str) -> EnvSpec:
    """Retrieve the :class:`EnvSpec` for the environment id from the :attr:`registry`.

    Args:
        env_id: The environment id with the expected format of ``[(namespace)/]id[-v(version)]``

    Returns:
        The environment spec if it exists

    Raises:
        Error: If the environment id doesn't exist
    """
    env_spec = registry.get(env_id)
    if env_spec is None:
        ns, name, version = parse_env_id(env_id)
        _check_version_exists(ns, name, version)
        raise error.Error(f"No registered env with id: {env_id}")
    else:
        assert isinstance(
            env_spec, EnvSpec
        ), f"Expected the registry for {env_id} to be an `EnvSpec`, actual type is {type(env_spec)}"
        return env_spec


def pprint_registry(
    print_registry: dict[str, EnvSpec] = registry,
    *,
    num_cols: int = 3,
    exclude_namespaces: list[str] | None = None,
    disable_print: bool = False,
) -> str | None:
    """Pretty prints all environments in the :attr:`registry`.

    Note:
        All arguments are keyword only

    Args:
        print_registry: Environment registry to be printed. By default, :attr:`registry`
        num_cols: Number of columns to arrange environments in, for display.
        exclude_namespaces: A list of namespaces to be excluded from printing. Helpful if only ALE environments are wanted.
        disable_print: Whether to return a string of all the namespaces and environment IDs
            or to print the string to console.
    """
    # Defaultdict to store environment ids according to namespace.
    namespace_envs: dict[str, list[str]] = defaultdict(list)
    max_justify = float("-inf")

    # Find the namespace associated with each environment spec
    for env_spec in print_registry.values():
        ns = env_spec.namespace

        if ns is None and isinstance(env_spec.entry_point, str):
            # Use regex to obtain namespace from entrypoints.
            env_entry_point = re.sub(r":\w+", "", env_spec.entry_point)
            split_entry_point = env_entry_point.split(".")

            if len(split_entry_point) >= 3:
                # If namespace is of the format:
                #  - gymnasium.envs.mujoco.ant_v4:AntEnv
                #  - gymnasium.envs.mujoco:HumanoidEnv
                ns = split_entry_point[2]
            elif len(split_entry_point) > 1:
                # If namespace is of the format - shimmy.atari_env
                ns = split_entry_point[1]
            else:
                # If namespace cannot be found, default to env name
                ns = env_spec.name

        namespace_envs[ns].append(env_spec.id)
        max_justify = max(max_justify, len(env_spec.name))

    # Iterate through each namespace and print environment alphabetically
    output: list[str] = []
    for ns, env_ids in namespace_envs.items():
        # Ignore namespaces to exclude.
        if exclude_namespaces is not None and ns in exclude_namespaces:
            continue

        # Print the namespace
        namespace_output = f"{'=' * 5} {ns} {'=' * 5}\n"

        # Reference: https://stackoverflow.com/a/33464001
        for count, env_id in enumerate(sorted(env_ids), 1):
            # Print column with justification.
            namespace_output += env_id.ljust(max_justify) + " "

            # Once all rows printed, switch to new column.
            if count % num_cols == 0:
                namespace_output = namespace_output.rstrip(" ")

                if count != len(env_ids):
                    namespace_output += "\n"

        output.append(namespace_output.rstrip(" "))

    if disable_print:
        return "\n".join(output)
    else:
        print("\n".join(output))
