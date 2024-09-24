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

# Initializing pygame initializes audio connections through SDL. SDL uses alsa by default on all Linux systems
# SDL connecting to alsa frequently create these giant lists of warnings every time you import an environment using
#   pygame
# DSP is far more benign (and should probably be the default in SDL anyways)

import os
import sys

if sys.platform.startswith("linux"):
    os.environ["SDL_AUDIODRIVER"] = "dsp"

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
__version__ = "1.0.0"

try:
    from farama_notifications import notifications

    if "gymnasium" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["gymnasium"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass
