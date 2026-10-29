"""Runs the Python code blocks in the documentation's markdown pages to prevent them drifting from the code.

For each page, the ```python blocks are run in order with a shared namespace (as a reader would run them).
Blocks containing ``>>>`` are run as doctests, so their outputs are checked, otherwise the block is executed.
To skip a block that is not meant to be run (e.g., the old Gym API), add ``<!-- test: skip -->`` on the line before it.
"""

from __future__ import annotations

import doctest
import importlib.util
import pathlib
import re
import sys

import pytest

import gymnasium as gym

DOCS_DIR = pathlib.Path(__file__).parent.parent / "docs"
DOC_PAGES = [
    *sorted((DOCS_DIR / "introduction").glob("*.md")),
    *sorted((DOCS_DIR / "api").rglob("*.md")),
    DOCS_DIR / "environments" / "mujoco.md",
]
SKIP_MARKER = "<!-- test: skip -->"
CODE_BLOCK_PATTERN = re.compile(
    r"(?P<marker>^<!-- test: skip -->\n)?^```python\n(?P<code>.*?)^```$",
    re.DOTALL | re.MULTILINE,
)


def get_code_blocks(page: pathlib.Path) -> list[tuple[int, str]]:
    """Returns the line number and code of each ```python block that is not skipped."""
    text = page.read_text()
    return [
        (text.count("\n", 0, match.start("code")) + 1, match.group("code"))
        for match in CODE_BLOCK_PATTERN.finditer(text)
        if match.group("marker") is None
    ]


@pytest.mark.parametrize(
    "page",
    [page for page in DOC_PAGES if get_code_blocks(page)],
    ids=lambda page: str(page.relative_to(DOCS_DIR)),
)
def test_docs_code(page: pathlib.Path, tmp_path, monkeypatch):
    """Runs the code blocks of a documentation page."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    # use a non-interactive backend so `plt.show()` doesn't block
    monkeypatch.setenv("MPLBACKEND", "Agg")
    if "matplotlib" in sys.modules:
        sys.modules["matplotlib"].use("Agg")

    namespace = {"__name__": "__main__"}
    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    for line_number, code in get_code_blocks(page):
        code = code.replace('render_mode="human"', 'render_mode="rgb_array"')
        name = f"{page.relative_to(DOCS_DIR)}:{line_number}"

        try:
            if ">>>" in code:
                test = doctest.DocTestParser().get_doctest(
                    code, namespace, name, str(page), line_number
                )
                runner.run(test, clear_globs=False)
                namespace.update(test.globs)
            else:
                exec(compile(code, name, "exec"), namespace)
        except gym.error.DependencyNotInstalled as e:
            pytest.skip(f"{name}: {e}")

    results = runner.summarize(verbose=False)
    assert results.failed == 0, f"{results.failed} doctest examples failed in {page}"


def test_skip_marker_is_used_before_code_blocks():
    """Checks that every skip marker is directly before a ```python block, such that it is not silently ignored."""
    for page in DOC_PAGES:
        lines = page.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.strip() == SKIP_MARKER:
                assert i + 1 < len(lines) and lines[i + 1] == "```python", (
                    f"{page}:{i + 1}: {SKIP_MARKER!r} must be on the line directly before a ```python block"
                )


def test_wrapper_table_is_up_to_date():
    """Checks that `docs/api/wrappers/table.md` matches the wrappers and their docstrings."""
    spec = importlib.util.spec_from_file_location(
        "gen_wrapper_table", DOCS_DIR / "_scripts" / "gen_wrapper_table.py"
    )
    gen_wrapper_table = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_wrapper_table)

    assert (DOCS_DIR / "api" / "wrappers" / "table.md").read_text() == (
        gen_wrapper_table.generate_page()
    ), (
        "The wrapper table is out of date, run `python docs/_scripts/gen_wrapper_table.py` to update it."
    )
