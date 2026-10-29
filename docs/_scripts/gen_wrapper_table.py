"""Generates `docs/api/wrappers/table.md` from the wrapper docstrings, run with `--check` to only check that the table is up to date."""

import inspect
import os.path
import re
import sys

import gymnasium as gym

exclude_wrappers = {"vector"}

TABLE_FILENAME = os.path.join(
    os.path.dirname(__file__), "..", "api", "wrappers", "table.md"
)


def wrapper_summary(wrapper: type) -> str:
    """Returns the first paragraph of the wrapper's docstring on a single line.

    As the table's current module is `gymnasium.wrappers`, references to `Env` and `Wrapper` are made absolute.
    """
    summary = " ".join(inspect.getdoc(wrapper).split("\n\n")[0].split())
    return re.sub(
        r":(class|meth|attr|func):`(Env|Wrapper)\b", r":\1:`gymnasium.\2", summary
    )


def generate_table(module, wrapper_names) -> str:
    table = ""
    for wrapper_name in sorted(wrapper_names):
        table += f"""    * - :class:`{wrapper_name}`
      - {wrapper_summary(getattr(module, wrapper_name))}
"""
    return table


def generate_page() -> str:
    wrapper_table = generate_table(
        gym.wrappers, set(gym.wrappers.__all__) - exclude_wrappers
    )
    vector_table = generate_table(
        gym.wrappers.vector,
        set(gym.wrappers.vector.__all__) - set(gym.wrappers.__all__),
    )

    return f"""# List of Wrappers

Gymnasium provides a number of commonly used wrappers listed below. More information can be found on the particular
wrapper in the page on the wrapper type

```{{eval-rst}}
.. py:currentmodule:: gymnasium.wrappers

.. list-table::
    :header-rows: 1

    * - Name
      - Description
{wrapper_table}```

## Vector Only Wrappers

```{{eval-rst}}
.. py:currentmodule:: gymnasium.wrappers.vector

.. list-table::
    :header-rows: 1

    * - Name
      - Description
{vector_table}```
"""


if __name__ == "__main__":
    page = generate_page()

    if "--check" in sys.argv:
        with open(TABLE_FILENAME) as file:
            if file.read() != page:
                sys.exit(
                    f"{TABLE_FILENAME} is out of date, run `python docs/_scripts/gen_wrapper_table.py` to update it."
                )
    else:
        with open(TABLE_FILENAME, "w") as file:
            file.write(page)
