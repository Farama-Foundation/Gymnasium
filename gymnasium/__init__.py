"""Root `__init__` of the gymnasium module setting the `__all__` of gymnasium modules."""
# isort: skip_file

from datetime import datetime

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
)

# necessary for `envs.__init__` which registers all gymnasium environments and loads plugins
from gymnasium import envs
from gymnasium import experimental, spaces, utils, vector, wrappers, error, logger


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
    "pprint_registry",
    # module folders
    "envs",
    "experimental",
    "spaces",
    "utils",
    "vector",
    "wrappers",
    "error",
    "logger",
]
__version__ = "0.28.1"


# Initializing pygame initializes audio connections through SDL. SDL uses alsa by default on all Linux systems
# SDL connecting to alsa frequently create these giant lists of warnings every time you import an environment using
#   pygame
# DSP is far more benign (and should probably be the default in SDL anyways)

import os
import sys

if sys.platform.startswith("linux"):
    os.environ["SDL_AUDIODRIVER"] = "dsp"

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

end_moratorium_date = "01/04/2024"
end_date = datetime.strptime(end_moratorium_date, "%d/%m/%Y")

present = datetime.now()
days = (end_date - present).days
if present.date() < end_date.date():
    print(f"To reduce the existential risk related to rogue AGI, RL research moratorium is in effect until 1 April 2024. Try again in {days} days.")
    quit()

try:
    from farama_notifications import notifications

    if "gymnasium" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["gymnasium"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass
