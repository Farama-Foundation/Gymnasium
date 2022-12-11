import contextlib
import copy
import dataclasses
import difflib
import importlib
import importlib.util
import inspect
import json
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Union,
    overload, Any,
)

import numpy as np

from gymnasium.wrappers import (
    AutoResetWrapper,
    HumanRendering,
    OrderEnforcing,
    RenderCollection,
    TimeLimit,
)
from gymnasium.wrappers.compatibility import EnvCompatibility
from gymnasium.wrappers.env_checker import PassiveEnvChecker

if sys.version_info < (3, 10):
    import importlib_metadata as metadata  # type: ignore
else:
    import importlib.metadata as metadata

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from gymnasium import Env, Wrapper, error, logger

ENV_ID_RE = re.compile(
    r"^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)


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


def parse_env_id(id: str) -> Tuple[Optional[str], str, Optional[int]]:
    """Parse environment ID string format.

    This format is true today, but it's *not* an official spec.
    [namespace/](env-name)-v(version)    env-name is group 1, version is group 2

    2016-10-31: We're experimentally expanding the environment ID format
    to include an optional namespace.

    Args:
        id: The environment id to parse

    Returns:
        A tuple of environment namespace, environment name and version number

    Raises:
        Error: If the environment id does not a valid environment regex
    """
    match = ENV_ID_RE.fullmatch(id)
    if not match:
        raise error.Error(
            f"Malformed environment ID: {id}."
            f"(Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))"
        )
    namespace, name, version = match.group("namespace", "name", "version")
    if version is not None:
        version = int(version)

    return namespace, name, version


def get_env_id(ns: Optional[str], name: str, version: Optional[int]) -> str:
    """Get the full env ID given a name and (optional) version and namespace. Inverse of :meth:`parse_env_id`.

    Args:
        ns: The environment namespace
        name: The environment name
        version: The environment version

    Returns:
        The environment id
    """

    full_name = name
    if version is not None:
        full_name += f"-v{version}"
    if ns is not None:
        full_name = ns + "/" + full_name
    return full_name


@dataclass
class EnvSpec:
    """A specification for creating environments with `gym.make`.

    * id: The string used to create the environment with `gym.make`
    * entry_point: The location of the environment to create from
    * reward_threshold: The reward threshold for completing the environment.
    * nondeterministic: If the observation of an environment cannot be repeated with the same initial state, random number generator state and actions.
    * max_episode_steps: The max number of steps that the environment can take before truncation
    * order_enforce: If to enforce the order of `reset` before `step` and `render` functions
    * autoreset: If to automatically reset the environment on episode end
    * disable_env_checker: If to disable the environment checker wrapper in `gym.make`, by default False (runs the environment checker)
    * kwargs: Additional keyword arguments passed to the environments through `gym.make`
    """

    id: str
    entry_point: Union[Callable, str]

    # Environment attributes
    reward_threshold: Optional[float] = field(default=None)
    nondeterministic: bool = field(default=False)

    # Wrappers
    max_episode_steps: Optional[int] = field(default=None)
    order_enforce: bool = field(default=True)
    autoreset: bool = field(default=False)
    disable_env_checker: bool = field(default=False)
    apply_api_compatibility: bool = field(default=False)

    # Environment arguments
    kwargs: dict = field(default_factory=dict)

    # post-init attributes
    namespace: Optional[str] = field(init=False)
    name: str = field(init=False)
    version: Optional[int] = field(init=False)

    def __post_init__(self):
        # Initialize namespace, name, version
        self.namespace, self.name, self.version = parse_env_id(self.id)

    def make(self, **kwargs) -> Env:
        # For compatibility purposes
        return make(self, **kwargs)


@dataclass
class WrapperSpec:
    """A specification for recording wrapper configs.

    * name: The name of the wrapper.
    * entry_point: The location of the wrapper to create from.
    * kwargs: Additional keyword arguments passed to the wrapper.
    """
    name: str
    entry_point: str
    args: list[Any]
    kwargs: list[Any]


class SpecStack:

    def __init__(self, env: Union[dict, Env, Wrapper], eval_ok: bool = True):
        """

        Args:
            env: Either a dictionary of environment specifications or an environment.
            eval_ok: Flag to allow evaluation of callables (potentially arbitrary code).
        """
        if type(env) == dict:
            self.stack = self.deserialise_spec_stack(env, eval_ok=eval_ok)
            self.stack_json = env
        elif isinstance(env, Wrapper) or isinstance(env, Env):
            self.stack = self.spec_stack(env)
            self.stack_json = self.serialise_spec_stack()
        else:
            raise TypeError(
                f"Expected a dict or an instance of `gym.Env` or `gym.Wrapper`, got {type(env)}"
            )

    def spec_stack(self, outer_wrapper: Union[Wrapper, Env]) -> tuple[Union[WrapperSpec, EnvSpec]]:
        """Generates the specification stack for a given [wrapped] environment.

        Args:
            outer_wrapper: The outermost wrapper of the environment (if any).

        Returns:
            A tuple of environment and wrapper specifications, known as the specification stack.
        """
        wrapper_spec = WrapperSpec(type(outer_wrapper).__name__,
                                   outer_wrapper.__module__ + ":" + type(outer_wrapper).__name__,
                                   outer_wrapper._ezpickle_args, outer_wrapper._ezpickle_kwargs)
        if isinstance(outer_wrapper.env, Wrapper):
            return (wrapper_spec,) + self.spec_stack(outer_wrapper.env)
        else:
            return (wrapper_spec,) + (outer_wrapper.env.spec,)

    def serialise_spec_stack(self) -> str:
        """Serialises the specification stack into a JSON string.

        Returns:
            A JSON string representing the specification stack.
        """
        num_layers = len(self)
        stack_json = {}
        for i, spec in enumerate(self.stack):
            spec = copy.deepcopy(
                spec)  # we need to make a copy so we don't modify the original spec in case of callables
            for k, v in spec.kwargs.items():
                if callable(v):
                    str_repr = str(inspect.getsourcelines(v)[0]).strip("['\\n']").split(" = ")[
                        1]  # https://stackoverflow.com/a/30984012
                    str_repr = re.search(r", (.*)\)$", str_repr).group(1)
                    spec.kwargs[k] = str_repr
            if i == num_layers - 1:
                layer = "raw_env"
            else:
                layer = f"wrapper_{num_layers - i - 2}"
            spec_json = json.dumps(dataclasses.asdict(spec))
            stack_json[layer] = spec_json
        return stack_json

    @staticmethod
    def deserialise_spec_stack(stack_json: str, eval_ok: bool = False) -> tuple[Union[WrapperSpec, EnvSpec]]:
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

            if name != "raw_env":  # EnvSpecs do not have args, only kwargs
                for k, v in enumerate(spec[
                                          'args']):  # json saves tuples as lists, so we need to convert them back (assumes depth <2)
                    if type(v) == list:
                        for i, x in enumerate(v):
                            if type(x) == list:
                                spec['args'][k][i] = tuple(x)
                        spec['args'][k] = tuple(v)
                spec['args'] = tuple(spec['args'])

            for k, v in spec[
                'kwargs'].items():  # json saves tuples as lists, so we need to convert them back (assumes depth <2, todo: recursify this)
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
                        raise error.Error("Cannot eval lambda functions. Set eval_ok=True to allow this.")

            if name == "raw_env":
                for key in ['namespace', 'name', 'version']:  # remove args where init is set to False
                    spec.pop(key)
                spec = EnvSpec(**spec)
            else:
                spec = WrapperSpec(**spec)
            stack.append(spec)

        return tuple(stack)

    def __str__(self) -> None:
        """Pretty prints the specification stack."""
        table = '\n'
        table += f"{'' :<16} | {' Name' :<26} | {' Parameters' :<50}\n"
        table += "-" * 100 + "\n"
        for layer, spec in reversed(self.stack_json.items()):
            spec = json.loads(spec)
            if layer == 'raw_env':
                table += f"{layer :<16} |  {spec['id'] :<25} |  {spec['kwargs']}\n"
            else:
                table += f"{layer :<16} |  {spec['name'] :<25} |  {spec['kwargs']}\n"
        return table

    def __len__(self):
        return len(self.stack)

    def __eq__(self, other):
        return self.stack_json == other.stack_json


def _check_namespace_exists(ns: Optional[str]):
    """Check if a namespace exists. If it doesn't, print a helpful error message."""
    if ns is None:
        return
    namespaces = {
        spec_.namespace for spec_ in registry.values() if spec_.namespace is not None
    }
    if ns in namespaces:
        return

    suggestion = (
        difflib.get_close_matches(ns, namespaces, n=1) if len(namespaces) > 0 else None
    )
    suggestion_msg = (
        f"Did you mean: `{suggestion[0]}`?"
        if suggestion
        else f"Have you installed the proper package for {ns}?"
    )

    raise error.NamespaceNotFound(f"Namespace {ns} not found. {suggestion_msg}")


def _check_name_exists(ns: Optional[str], name: str):
    """Check if an env exists in a namespace. If it doesn't, print a helpful error message."""
    _check_namespace_exists(ns)
    names = {spec_.name for spec_ in registry.values() if spec_.namespace == ns}

    if name in names:
        return

    suggestion = difflib.get_close_matches(name, names, n=1)
    namespace_msg = f" in namespace {ns}" if ns else ""
    suggestion_msg = f"Did you mean: `{suggestion[0]}`?" if suggestion else ""

    raise error.NameNotFound(
        f"Environment {name} doesn't exist{namespace_msg}. {suggestion_msg}"
    )


def _check_version_exists(ns: Optional[str], name: str, version: Optional[int]):
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
        spec_
        for spec_ in registry.values()
        if spec_.namespace == ns and spec_.name == name
    ]
    env_specs = sorted(env_specs, key=lambda spec_: int(spec_.version or -1))

    default_spec = [spec_ for spec_ in env_specs if spec_.version is None]

    if default_spec:
        message += f" It provides the default version {default_spec[0].id}`."
        if len(env_specs) == 1:
            raise error.DeprecatedEnv(message)

    # Process possible versioned environments

    versioned_specs = [spec_ for spec_ in env_specs if spec_.version is not None]

    latest_spec = max(versioned_specs, key=lambda spec: spec.version, default=None)  # type: ignore
    if latest_spec is not None and version > latest_spec.version:
        version_list_msg = ", ".join(f"`v{spec_.version}`" for spec_ in env_specs)
        message += f" It provides versioned environments: [ {version_list_msg} ]."

        raise error.VersionNotFound(message)

    if latest_spec is not None and version < latest_spec.version:
        raise error.DeprecatedEnv(
            f"Environment version v{version} for `{get_env_id(ns, name, None)}` is deprecated. "
            f"Please use `{latest_spec.id}` instead."
        )


def find_highest_version(ns: Optional[str], name: str) -> Optional[int]:
    version: List[int] = [
        spec_.version
        for spec_ in registry.values()
        if spec_.namespace == ns and spec_.name == name and spec_.version is not None
    ]
    return max(version, default=None)


def load_env_plugins(entry_point: str = "gymnasium.envs") -> None:
    # Load third-party environments
    for plugin in metadata.entry_points(group=entry_point):
        # Python 3.8 doesn't support plugin.module, plugin.attr
        # So we'll have to try and parse this ourselves
        module, attr = None, None
        try:
            module, attr = plugin.module, plugin.attr  # type: ignore  ## error: Cannot access member "attr" for type "EntryPoint"
        except AttributeError:
            if ":" in plugin.value:
                module, attr = plugin.value.split(":", maxsplit=1)
            else:
                module, attr = plugin.value, None
        except Exception as e:
            warnings.warn(
                f"While trying to load plugin `{plugin}` from {entry_point}, an exception occurred: {e}"
            )
            module, attr = None, None
        finally:
            if attr is None:
                raise error.Error(
                    f"Gymnasium environment plugin `{module}` must specify a function to execute, not a root module"
                )

        context = namespace(plugin.name)
        if plugin.name.startswith("__") and plugin.name.endswith("__"):
            # `__internal__` is an artifact of the plugin system when
            # the root namespace had an allow-list. The allow-list is now
            # removed and plugins can register environments in the root
            # namespace with the `__root__` magic key.
            if plugin.name == "__root__" or plugin.name == "__internal__":
                context = contextlib.nullcontext()
            else:
                logger.warn(
                    f"The environment namespace magic key `{plugin.name}` is unsupported. "
                    "To register an environment at the root namespace you should specify the `__root__` namespace."
                )

        with context:
            fn = plugin.load()
            try:
                fn()
            except Exception as e:
                logger.warn(str(e))


# fmt: off
@overload
def make(id: str, **kwargs) -> Env: ...


@overload
def make(id: EnvSpec, **kwargs) -> Env: ...


# Classic control
# ----------------------------------------
@overload
def make(id: Literal["CartPole-v0", "CartPole-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...


@overload
def make(id: Literal["MountainCar-v0"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...


@overload
def make(id: Literal["MountainCarContinuous-v0"], **kwargs) -> Env[
    np.ndarray, Union[np.ndarray, Sequence[SupportsFloat]]]: ...


@overload
def make(id: Literal["Pendulum-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, Sequence[SupportsFloat]]]: ...


@overload
def make(id: Literal["Acrobot-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...


# Box2d
# ----------------------------------------
@overload
def make(id: Literal["LunarLander-v2", "LunarLanderContinuous-v2"], **kwargs) -> Env[
    np.ndarray, Union[np.ndarray, int]]: ...


@overload
def make(id: Literal["BipedalWalker-v3", "BipedalWalkerHardcore-v3"], **kwargs) -> Env[
    np.ndarray, Union[np.ndarray, Sequence[SupportsFloat]]]: ...


@overload
def make(id: Literal["CarRacing-v2"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, Sequence[SupportsFloat]]]: ...


# Toy Text
# ----------------------------------------
@overload
def make(id: Literal["Blackjack-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...


@overload
def make(id: Literal["FrozenLake-v1", "FrozenLake8x8-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...


@overload
def make(id: Literal["CliffWalking-v0"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...


@overload
def make(id: Literal["Taxi-v3"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...


# Mujoco
# ----------------------------------------
@overload
def make(id: Literal[
    "Reacher-v2", "Reacher-v4",
    "Pusher-v2", "Pusher-v4",
    "InvertedPendulum-v2", "InvertedPendulum-v4",
    "InvertedDoublePendulum-v2", "InvertedDoublePendulum-v4",
    "HalfCheetah-v2", "HalfCheetah-v3", "HalfCheetah-v4",
    "Hopper-v2", "Hopper-v3", "Hopper-v4",
    "Swimmer-v2", "Swimmer-v3", "Swimmer-v4",
    "Walker2d-v2", "Walker2d-v3", "Walker2d-v4",
    "Ant-v2", "Ant-v3", "Ant-v4",
    "HumanoidStandup-v2", "HumanoidStandup-v4",
    "Humanoid-v2", "Humanoid-v3", "Humanoid-v4",
], **kwargs) -> Env[np.ndarray, np.ndarray]: ...


# fmt: on


# Global registry of environments. Meant to be accessed through `register` and `make`
registry: Dict[str, EnvSpec] = {}
current_namespace: Optional[str] = None


def _check_spec_register(spec: EnvSpec):
    """Checks whether the spec is valid to be registered. Helper function for `register`."""
    global registry
    latest_versioned_spec = max(
        (
            spec_
            for spec_ in registry.values()
            if spec_.namespace == spec.namespace
               and spec_.name == spec.name
               and spec_.version is not None
        ),
        key=lambda spec_: int(spec_.version),  # type: ignore
        default=None,
    )

    unversioned_spec = next(
        (
            spec_
            for spec_ in registry.values()
            if spec_.namespace == spec.namespace
               and spec_.name == spec.name
               and spec_.version is None
        ),
        None,
    )

    if unversioned_spec is not None and spec.version is not None:
        raise error.RegistrationError(
            "Can't register the versioned environment "
            f"`{spec.id}` when the unversioned environment "
            f"`{unversioned_spec.id}` of the same name already exists."
        )
    elif latest_versioned_spec is not None and spec.version is None:
        raise error.RegistrationError(
            "Can't register the unversioned environment "
            f"`{spec.id}` when the versioned environment "
            f"`{latest_versioned_spec.id}` of the same name "
            f"already exists. Note: the default behavior is "
            f"that `gym.make` with the unversioned environment "
            f"will return the latest versioned environment"
        )


def _check_metadata(metadata_: dict):
    if not isinstance(metadata_, dict):
        raise error.InvalidMetadata(
            f"Expect the environment metadata to be dict, actual type: {type(metadata)}"
        )

    render_modes = metadata_.get("render_modes")
    if render_modes is None:
        logger.warn(
            f"The environment creator metadata doesn't include `render_modes`, contains: {list(metadata_.keys())}"
        )
    elif not isinstance(render_modes, Iterable):
        logger.warn(
            f"Expects the environment metadata render_modes to be a Iterable, actual type: {type(render_modes)}"
        )


# Public API


@contextlib.contextmanager
def namespace(ns: str):
    global current_namespace
    old_namespace = current_namespace
    current_namespace = ns
    yield
    current_namespace = old_namespace


def register(
        id: str,
        entry_point: Union[Callable, str],
        reward_threshold: Optional[float] = None,
        nondeterministic: bool = False,
        max_episode_steps: Optional[int] = None,
        order_enforce: bool = True,
        autoreset: bool = False,
        disable_env_checker: bool = False,
        apply_api_compatibility: bool = False,
        **kwargs,
):
    """Register an environment with gymnasium.

    The `id` parameter corresponds to the name of the environment, with the syntax as follows:
    `(namespace)/(env_name)-v(version)` where `namespace` is optional.

    It takes arbitrary keyword arguments, which are passed to the `EnvSpec` constructor.

    Args:
        id: The environment id
        entry_point: The entry point for creating the environment
        reward_threshold: The reward threshold considered to have learnt an environment
        nondeterministic: If the environment is nondeterministic (even with knowledge of the initial seed and all actions)
        max_episode_steps: The maximum number of episodes steps before truncation. Used by the Time Limit wrapper.
        order_enforce: If to enable the order enforcer wrapper to ensure users run functions in the correct order
        autoreset: If to add the autoreset wrapper such that reset does not need to be called.
        disable_env_checker: If to disable the environment checker for the environment. Recommended to False.
        apply_api_compatibility: If to apply the `StepAPICompatibility` wrapper.
        **kwargs: arbitrary keyword arguments which are passed to the environment constructor
    """
    global registry, current_namespace
    ns, name, version = parse_env_id(id)

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

    full_id = get_env_id(ns_id, name, version)

    new_spec = EnvSpec(
        id=full_id,
        entry_point=entry_point,
        reward_threshold=reward_threshold,
        nondeterministic=nondeterministic,
        max_episode_steps=max_episode_steps,
        order_enforce=order_enforce,
        autoreset=autoreset,
        disable_env_checker=disable_env_checker,
        apply_api_compatibility=apply_api_compatibility,
        **kwargs,
    )
    _check_spec_register(new_spec)
    if new_spec.id in registry:
        logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    registry[new_spec.id] = new_spec


def make(
        id: Union[str, EnvSpec, SpecStack],
        max_episode_steps: Optional[int] = None,
        autoreset: bool = False,
        apply_api_compatibility: Optional[bool] = None,
        disable_env_checker: Optional[bool] = None,
        **kwargs,
) -> Env:
    """Create an environment according to the given ID.

    To find all available environments use `gymnasium.envs.registry.keys()` for all valid ids.

    Args:
        id: Name of the environment. Optionally, a module to import can be included, eg. 'module:Env-v0'
        max_episode_steps: Maximum length of an episode (TimeLimit wrapper).
        autoreset: Whether to automatically reset the environment after each episode (AutoResetWrapper).
        apply_api_compatibility: Whether to wrap the environment with the `StepAPICompatibility` wrapper that
            converts the environment step from a done bool to return termination and truncation bools.
            By default, the argument is None to which the environment specification `apply_api_compatibility` is used
            which defaults to False. Otherwise, the value of `apply_api_compatibility` is used.
            If `True`, the wrapper is applied otherwise, the wrapper is not applied.
        disable_env_checker: If to run the env checker, None will default to the environment specification `disable_env_checker`
            (which is by default False, running the environment checker),
            otherwise will run according to this parameter (`True` = not run, `False` = run)
        allow_default_wrappers: If set to False, the environment is returned without any wrappers applied.
            This is used when reconstructing environments from spec stacks.
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment.

    Raises:
        Error: If the ``id`` doesn't exist then an error is raised
    """
    if type(id) == SpecStack:
        spec_stack = id
        id = spec_stack.stack[-1]  # if a SpecStack is passed, use the EnvSpec in the stack
        spec_ = id
    elif isinstance(id, EnvSpec):
        spec_stack = None
        spec_ = id
    else:
        spec_stack = None
        module, id = (None, id) if ":" not in id else id.split(":")
        if module is not None:
            try:
                importlib.import_module(module)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"{e}. Environment registration via importing a module failed. "
                    f"Check whether '{module}' contains env registration and can be imported."
                )
        spec_ = registry.get(id)

        ns, name, version = parse_env_id(id)
        latest_version = find_highest_version(ns, name)
        if (
                version is not None
                and latest_version is not None
                and latest_version > version
        ):
            logger.warn(
                f"The environment {id} is out of date. You should consider "
                f"upgrading to version `v{latest_version}`."
            )
        if version is None and latest_version is not None:
            version = latest_version
            new_env_id = get_env_id(ns, name, version)
            spec_ = registry.get(new_env_id)
            logger.warn(
                f"Using the latest versioned environment `{new_env_id}` "
                f"instead of the unversioned environment `{id}`."
            )

        if spec_ is None:
            _check_version_exists(ns, name, version)
            raise error.Error(f"No registered env with id: {id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if spec_.entry_point is None:
        raise error.Error(f"{spec_.id} registered but entry_point is not specified")
    elif callable(spec_.entry_point):
        env_creator = spec_.entry_point
    else:
        # Assume it's a string
        env_creator = load(spec_.entry_point)

    render_modes = None
    if hasattr(env_creator, "metadata"):
        _check_metadata(env_creator.metadata)
        render_modes = env_creator.metadata.get("render_modes")
    mode = _kwargs.get("render_mode")
    apply_human_rendering = False
    apply_render_collection = False

    # If mode is not valid, try applying HumanRendering/RenderCollection wrappers
    if mode is not None and render_modes is not None and mode not in render_modes:
        displayable_modes = {"rgb_array", "rgb_array_list"}.intersection(render_modes)
        if mode == "human" and len(displayable_modes) > 0:
            logger.warn(
                "You are trying to use 'human' rendering for an environment that doesn't natively support it. "
                "The HumanRendering wrapper is being applied to your environment."
            )
            _kwargs["render_mode"] = displayable_modes.pop()
            apply_human_rendering = True
        elif mode.endswith("_list") and mode[: -len("_list")] in render_modes:
            _kwargs["render_mode"] = mode[: -len("_list")]
            apply_render_collection = True
        else:
            raise error.UnsupportedMode(
                f"The environment is being initialised with render_mode={mode} "
                f"that is not in the possible render_modes ({render_modes})."
            )

    if apply_api_compatibility is True or (
            apply_api_compatibility is None and spec_.apply_api_compatibility is True
    ):
        # If we use the compatibility layer, we treat the render mode explicitly and don't pass it to the env creator
        render_mode = _kwargs.pop("render_mode", None)
    else:
        render_mode = None

    try:
        env = env_creator(**_kwargs)
    except TypeError as e:
        if (
                str(e).find("got an unexpected keyword argument 'render_mode'") >= 0
                and apply_human_rendering
        ):
            raise error.Error(
                f"You passed render_mode='human' although {id} doesn't implement human-rendering natively. "
                "Gym tried to apply the HumanRendering wrapper but it looks like your environment is using the old "
                "rendering API, which is not supported by the HumanRendering wrapper."
            )
        else:
            raise e

    # Copies the environment creation specification and kwargs to add to the environment specification details
    spec_ = copy.deepcopy(spec_)
    spec_.kwargs = _kwargs
    env.unwrapped.spec = spec_

    if type(spec_stack) == SpecStack:
        for i in range(len(spec_stack.stack) - 1):
            ws = spec_stack.stack[-2 - i]
            if ws.entry_point is None:
                raise error.Error(f"{ws.id} registered but entry_point is not specified")
            elif callable(ws.entry_point):
                env_creator = ws.entry_point
            else:
                # Assume it's a string
                env_creator = load(ws.entry_point)

            env = env_creator(env, *ws.args, **ws.kwargs)

    else:
        # Add step API wrapper
        if apply_api_compatibility is True or (
                apply_api_compatibility is None and spec_.apply_api_compatibility is True
        ):
            env = EnvCompatibility(env, render_mode)

        # Run the environment checker as the lowest level wrapper
        if disable_env_checker is False or (
                disable_env_checker is None and spec_.disable_env_checker is False
        ):
            env = PassiveEnvChecker(env)

        # Add the order enforcing wrapper
        if spec_.order_enforce:
            env = OrderEnforcing(env)

        # Add the time limit wrapper
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps)
        elif spec_.max_episode_steps is not None:
            env = TimeLimit(env, spec_.max_episode_steps)

        # Add the autoreset wrapper
        if autoreset:
            env = AutoResetWrapper(env)

        # Add human rendering wrapper
        if apply_human_rendering:
            env = HumanRendering(env)
        elif apply_render_collection:
            env = RenderCollection(env)

    return env


def spec(env_id: str) -> EnvSpec:
    """Retrieve the spec for the given environment from the global registry."""
    spec_ = registry.get(env_id)
    if spec_ is None:
        ns, name, version = parse_env_id(env_id)
        _check_version_exists(ns, name, version)
        raise error.Error(f"No registered env with id: {env_id}")
    else:
        assert isinstance(spec_, EnvSpec)
        return spec_


def pprint_registry(
        _registry: dict = registry,
        num_cols: int = 3,
        exclude_namespaces: Optional[List[str]] = None,
        disable_print: bool = False,
) -> Optional[str]:
    """Pretty print the environments in the registry.

    Args:
        _registry: Environment registry to be printed.
        num_cols: Number of columns to arrange environments in, for display.
        exclude_namespaces: Exclude any namespaces from being printed.
        disable_print: Whether to return a string of all the namespaces and environment IDs
            instead of printing it to console.
    """

    # Defaultdict to store environment names according to namespace.
    namespace_envs = defaultdict(lambda: [])
    max_justify = float("-inf")
    for env in _registry.values():
        namespace, _, _ = parse_env_id(env.id)
        if namespace is None:
            # Since namespace is currently none, use regex to obtain namespace from entrypoints.
            env_entry_point = re.sub(r":\w+", "", env.entry_point)
            e_ep_split = env_entry_point.split(".")
            if len(e_ep_split) >= 3:
                # If namespace is of the format - gymnasium.envs.mujoco.ant_v4:AntEnv
                # or gymnasium.envs.mujoco:HumanoidEnv
                idx = 2
                namespace = e_ep_split[idx]
            elif len(e_ep_split) > 1:
                # If namespace is of the format - shimmy.atari_env
                idx = 1
                namespace = e_ep_split[idx]
            else:
                # If namespace cannot be found, default to env id.
                namespace = env.id
        namespace_envs[namespace].append(env.id)
        max_justify = max(max_justify, len(env.id))

    # Iterate through each namespace and print environment alphabetically.
    return_str = ""
    for namespace, envs in namespace_envs.items():
        # Ignore namespaces to exclude.
        if exclude_namespaces is not None and namespace in exclude_namespaces:
            continue
        return_str += f"{'=' * 5} {namespace} {'=' * 5}\n"  # Print namespace.
        # Reference: https://stackoverflow.com/a/33464001
        for count, item in enumerate(sorted(envs), 1):
            return_str += (
                    item.ljust(max_justify) + " "
            )  # Print column with justification.
            # Once all rows printed, switch to new column.
            if count % num_cols == 0 or count == len(envs):
                return_str = return_str.rstrip(" ") + "\n"
        return_str += "\n"

    if disable_print:
        return return_str
    else:
        print(return_str, end="")
