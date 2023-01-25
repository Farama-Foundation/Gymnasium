"""A set of common utilities used within the environments.

These are not intended as API functions, and will not remain stable over time.
"""

from gymnasium.utils import play, seeding
from gymnasium.utils.colorize import colorize
from gymnasium.utils.env_checker import check_env
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.utils.save_video import capped_cubic_video_schedule, save_video
from gymnasium.utils.step_api_compatibility import (
    convert_to_done_step_api,
    convert_to_terminated_truncated_step_api,
    step_api_compatibility,
)


__all__ = [
    # Files
    "seeding",
    "play",  # due to the core function being called play as well, then we cannot shorten the link.
    # Functions
    "colorize",
    "EzPickle",
    "check_env",
    "save_video",
    "capped_cubic_video_schedule",
    "step_api_compatibility",
    "convert_to_terminated_truncated_step_api",
    "convert_to_done_step_api",
]
