import os.path

import gymnasium as gym


exclude_wrappers = {"vector"}


def generate_wrappers():
    wrapper_table = ""
    for wrapper_name in sorted(gym.wrappers.__all__):
        if wrapper_name not in exclude_wrappers:
            wrapper_doc = getattr(gym.wrappers, wrapper_name).__doc__.split("\n")[0]
            wrapper_table += f"""    * - :class:`{wrapper_name}`
      - {wrapper_doc}
"""
    return wrapper_table


def generate_vector_wrappers():
    unique_vector_wrappers = set(gym.wrappers.vector.__all__) - set(
        gym.wrappers.__all__
    )

    vector_table = ""
    for vector_name in sorted(unique_vector_wrappers):
        vector_doc = getattr(gym.wrappers.vector, vector_name).__doc__.split("\n")[0]
        vector_table += f"""    * - :class:`{vector_name}`
      - {vector_doc}
"""
    return vector_table


if __name__ == "__main__":
    gen_wrapper_table = generate_wrappers()
    gen_vector_table = generate_vector_wrappers()

    page = f"""
# List of Gymnasium Wrappers

Gymnasium provides a number of commonly used wrappers listed below. More information can be found on the particular
wrapper in the page on the wrapper type

```{{eval-rst}}
.. py:currentmodule:: gymnasium.wrappers

.. list-table::
    :header-rows: 1

    * - Name
      - Description
{gen_wrapper_table}
```

## Vector only Wrappers

```{{eval-rst}}
.. py:currentmodule:: gymnasium.wrappers.vector

.. list-table::
    :header-rows: 1

    * - Name
      - Description
{gen_vector_table}
```
"""

    filename = os.path.join(
        os.path.dirname(__file__), "..", "api", "wrappers", "table.md"
    )
    with open(filename, "w") as file:
        file.write(page)
