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

These environments all involve toy games based around physics control, using [box2d](https://box2d.org/) based physics and PyGame-based rendering. These environments were contributed back in the early days of OpenAI Gym by Oleg Klimov, and have become popular toy benchmarks ever since. All environments are highly configurable via arguments specified in each environment's documentation.

The unique dependencies for this set of environments can be installed via:

````bash
pip install swig
pip install gymnasium[box2d]
````

[SWIG](https://swig.org/) is necessary for building the wheel for [box2d-py](https://pypi.org/project/box2d-py/), the Python package that provides bindings to box2d.
