"""A set of common utilities used within the environments.

These are not intended as API functions, and will not remain stable over time.
"""

# These submodules should not have any import-time dependencies.
# We want this since we use `utils` during our import-time sanity checks
# that verify that our dependencies are actually present.
from gymnasium.utils.colorize import colorize
from gymnasium.utils.env_checker import check_env
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.utils.play import PlayableGame, PlayPlot, play
from gymnasium.utils.save_video import capped_cubic_video_schedule, save_video
from gymnasium.utils.step_api_compatibility import (
    convert_to_done_step_api,
    convert_to_terminated_truncated_step_api,
    step_api_compatibility,
)
