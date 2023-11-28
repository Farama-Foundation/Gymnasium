"""Set of Error classes for gymnasium."""


class Error(Exception):
    """Error superclass."""


# Registration errors
class UnregisteredEnv(Error):
    """Raised when the user requests an env from the registry that does not actually exist."""


class NamespaceNotFound(UnregisteredEnv):
    """Raised when the user requests an env from the registry where the namespace doesn't exist."""


class NameNotFound(UnregisteredEnv):
    """Raised when the user requests an env from the registry where the name doesn't exist."""


class VersionNotFound(UnregisteredEnv):
    """Raised when the user requests an env from the registry where the version doesn't exist."""


class DeprecatedEnv(Error):
    """Raised when the user requests an env from the registry with an older version number than the latest env with the same name."""


class RegistrationError(Error):
    """Raised when the user attempts to register an invalid env. For example, an unversioned env when a versioned env exists."""


# Environment errors
class DependencyNotInstalled(Error):
    """Raised when the user has not installed a dependency."""


class UnsupportedMode(Error):
    """Raised when the user requests a rendering mode not supported by the environment."""


class InvalidMetadata(Error):
    """Raised when the metadata of an environment is not valid."""


class ResetNeeded(Error):
    """When the order enforcing is violated, i.e. step or render is called before reset."""


class InvalidAction(Error):
    """Raised when the user performs an action not contained within the action space."""


class MissingArgument(Error):
    """Raised when a required argument in the initializer is missing."""


class InvalidProbability(Error):
    """Raised when given an invalid value for a probability."""


class InvalidBound(Error):
    """Raised when the clipping an array with invalid upper and/or lower bound."""


# Wrapper errors
class DeprecatedWrapper(ImportError):
    """Error message for importing an old version of a wrapper."""


# Vectorized environments errors
class AlreadyPendingCallError(Exception):
    """Raised when `reset`, or `step` is called asynchronously (e.g. with `reset_async`, or `step_async` respectively), and `reset_async`, or `step_async` (respectively) is called again (without a complete call to `reset_wait`, or `step_wait` respectively)."""

    def __init__(self, message: str, name: str):
        """Initialises the exception with name attributes."""
        super().__init__(message)
        self.name = name


class NoAsyncCallError(Exception):
    """Raised when an asynchronous `reset`, or `step` is not running, but `reset_wait`, or `step_wait` (respectively) is called."""

    def __init__(self, message: str, name: str):
        """Initialises the exception with name attributes."""
        super().__init__(message)
        self.name = name


class ClosedEnvironmentError(Exception):
    """Trying to call `reset`, or `step`, while the environment is closed."""


class CustomSpaceError(Exception):
    """The space is a custom gymnasium.Space instance, and is not supported by `AsyncVectorEnv` with `shared_memory=True`."""
