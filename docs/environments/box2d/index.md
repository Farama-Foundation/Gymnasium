---
firstpage:
lastpage:
---

## Box2D

```{toctree}
:hidden:

bipedal_walker
car_racing
lunar_lander
``` 
  
```{raw} html
   :file: index.html
```
   
These environments all involve toy games based around physics control, using [box2d](https://box2d.org/) based physics and PyGame based rendering. These environments were contributed back in the early days of Gymnasium by Oleg Klimov, and have become popular toy benchmarks ever since. All environments are highly configurable via arguments specified in each environment's documentation.
  
The unique dependencies for this set of environments can be installed via:

````bash
pip install gymnasium[box2d]
````
