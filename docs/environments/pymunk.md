---
firstpage:
lastpage:
---

# Pymunk

```{toctree}
:hidden:

pymunk/lunar_lander
```

```{raw} html
   :file: pymunk/list.html
```

These environments use [Pymunk](https://www.pymunk.org/), a Pythonic 2D physics
library built on Chipmunk2D.

The unique dependencies for this set of environments can be installed with:

````bash
pip install gymnasium[pymunk]
````

`LunarLander-v4` and `LunarLanderContinuous-v4` use the Pymunk implementation.
The corresponding `v3` environments remain available in the Box2D family for
reproducibility.
