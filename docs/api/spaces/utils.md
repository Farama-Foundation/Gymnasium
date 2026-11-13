---
title: Utils
---

# Spaces Utils

Flattening a `OneOf` space supports empty subspaces. Padding uses the combined
space's dtype, including when the empty subspace has an integer dtype and another
subspace provides fractional or negative bounds.

```{eval-rst}
.. autofunction:: gymnasium.spaces.utils.flatten_space
.. autofunction:: gymnasium.spaces.utils.flatten
.. autofunction:: gymnasium.spaces.utils.flatdim
.. autofunction:: gymnasium.spaces.utils.unflatten
.. autofunction:: gymnasium.spaces.utils.is_space_dtype_shape_equiv
```
