"""Set of functions for logging messages."""

import warnings
from typing import Optional, Type

from gymnasium.utils import colorize


WARN = 30
ERROR = 40

min_level = 30


# Ensure DeprecationWarning to be displayed (#2685, #3059)
warnings.filterwarnings("once", "", DeprecationWarning, module=r"^gymnasium\.")


def warn(
    msg: str,
    *args: object,
    category: Optional[Type[Warning]] = None,
    stacklevel: int = 1,
):
    """Raises a warning to the user if the min_level <= WARN.

    Args:
        msg: The message to warn the user
        *args: Additional information to warn the user
        category: The category of warning
        stacklevel: The stack level to raise to
    """
    if min_level <= WARN:
        warnings.warn(
            colorize(f"WARN: {msg % args}", "yellow"),
            category=category,
            stacklevel=stacklevel + 1,
        )


def deprecation(msg: str, *args: object):
    """Logs a deprecation warning to users."""
    warn(msg, *args, category=DeprecationWarning, stacklevel=2)


def error(msg: str, *args: object):
    """Logs an error message if min_level <= ERROR in red on the sys.stderr."""
    if min_level <= ERROR:
        warnings.warn(colorize(f"ERROR: {msg % args}", "red"), stacklevel=3)
