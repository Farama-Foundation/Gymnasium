"""Set of functions for logging messages."""
from __future__ import annotations

import sys
import warnings


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

min_level = 30


# Ensure DeprecationWarning to be displayed (#2685, #3059)
warnings.filterwarnings("once", "", DeprecationWarning, module=r"^gymnasium\.")


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(
    string: str, color: str, bold: bool = False, highlight: bool = False
) -> str:
    """Returns string surrounded by appropriate terminal colour codes to print colourised text.

    Args:
        string: The message to colourise
        color: Literal values are gray, red, green, yellow, blue, magenta, cyan, white, crimson
        bold: If to bold the string
        highlight: If to highlight the string

    Returns:
        Colourised string
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return f"\x1b[{attrs}m{string}\x1b[0m"


def set_level(level: int):
    """Set logging threshold on current logger."""
    global min_level
    min_level = level


def debug(msg: str, *args: object):
    """Logs a debug message to the user."""
    if min_level <= DEBUG:
        print(f"DEBUG: {msg % args}", file=sys.stderr)


def info(msg: str, *args: object):
    """Logs an info message to the user."""
    if min_level <= INFO:
        print(f"INFO: {msg % args}", file=sys.stderr)


def warn(
    msg: str,
    *args: object,
    category: type[Warning] | None = None,
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
        print(colorize(f"ERROR: {msg % args}", "red"), file=sys.stderr)


# DEPRECATED:
setLevel = set_level
