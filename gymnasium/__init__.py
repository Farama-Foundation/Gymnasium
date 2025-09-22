"""Root `__init__` of the gymnasium module setting the `__all__` of gymnasium modules."""

# isort: skip_file

from gymnasium.core import (
    Env,
    Wrapper,
    ObservationWrapper,
    ActionWrapper,
    RewardWrapper,
)
from gymnasium.spaces.space import Space
from gymnasium.envs.registration import (
    make,
    spec,
    register,
    registry,
    pprint_registry,
    make_vec,
    VectorizeMode,
    register_envs,
)
from gymnasium import spaces, utils, vector, wrappers, error, logger, experimental

import os
import sys

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# necessary for `envs.__init__` which registers all gymnasium environments and loads plugins
from gymnasium import envs  # noqa: E402


__all__ = [
    # core classes
    "Env",
    "Wrapper",
    "ObservationWrapper",
    "ActionWrapper",
    "RewardWrapper",
    "Space",
    # registration
    "make",
    "make_vec",
    "spec",
    "register",
    "registry",
    "VectorizeMode",
    "pprint_registry",
    "register_envs",
    # module folders
    "envs",
    "spaces",
    "utils",
    "vector",
    "wrappers",
    "error",
    "logger",
    "experimental",
]
__version__ = "1.2.1"

try:
    from farama_notifications import notifications

    if "gymnasium" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["gymnasium"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass
