"""Run Sphinx's linkcheck builder against the docs.

Usage:
    python docs/_scripts/linkcheck.py [extra sphinx-build args]

Exits with sphinx-build's return code so it can be wired into CI.
"""

import os
import subprocess
import sys

DOCS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(os.path.dirname(DOCS_DIR), "_build", "linkcheck")


if __name__ == "__main__":
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "linkcheck",
        DOCS_DIR,
        BUILD_DIR,
        *sys.argv[1:],
    ]
    sys.exit(subprocess.call(cmd))