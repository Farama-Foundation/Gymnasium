---
firstpage:
lastpage:
---

# Box2D

```{toctree}
:hidden:

box2d/bipedal_walker
box2d/car_racing
box2d/lunar_lander
```

```{raw} html
   :file: box2d/list.html
```

```{warning}
If you plan to use box2d with Python version >= 3.14, know that at the moment the upstream project [pybox2d](https://github.com/pybox2d/pybox2d) does not provide pre-built wheels for >= 3.13. As a workaround, the `box2d` extra builds [box2d-py](https://pypi.org/project/box2d-py/) from source and additionally requires `swig`. Make sure you have [SWIG](https://swig.org/) installed on your system.
```

These environments all involve toy games based around physics control, using [box2d](https://box2d.org/) based physics and PyGame-based rendering. These environments were contributed back in the early days of OpenAI Gym by Oleg Klimov, and have become popular toy benchmarks ever since. All environments are highly configurable via arguments specified in each environment's documentation.

The unique dependencies for this set of environments can be installed via:

````bash
pip install swig
pip install gymnasium[box2d]
````

[SWIG](https://swig.org/) is necessary for building the wheel for [box2d-py](https://pypi.org/project/box2d-py/), the Python package that provides bindings to box2d on Python 3.14+. For Python 3.13 or below we use the pre-built wheel from [pybox2d](https://github.com/pybox2d/pybox2d).
