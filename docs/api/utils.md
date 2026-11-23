---
title: Utility functions
---

# Utility functions

## Seeding

```{eval-rst}
.. autofunction:: gymnasium.utils.seeding.np_random
```

## Environment checking

```{eval-rst}
.. autofunction:: gymnasium.utils.env_checker.check_env
.. autofunction:: gymnasium.utils.env_match.check_environments_match
```

## Visualization

```{eval-rst}
.. autofunction:: gymnasium.utils.play.play
.. autoclass:: gymnasium.utils.play.PlayPlot

    .. automethod:: callback

.. autoclass:: gymnasium.utils.play.PlayableGame

    .. automethod:: process_event
```

## Environment pickling

```{eval-rst}
.. autoclass:: gymnasium.utils.ezpickle.EzPickle
.. autoclass:: gymnasium.utils.RecordConstructorArgs
```

## Save rendering videos

```{eval-rst}
.. autofunction:: gymnasium.utils.save_video.save_video
.. autofunction:: gymnasium.utils.save_video.capped_cubic_video_schedule
```

## Old to new step API compatibility

```{eval-rst}
.. autofunction:: gymnasium.utils.step_api_compatibility.step_api_compatibility
.. autofunction:: gymnasium.utils.step_api_compatibility.convert_to_terminated_truncated_step_api
.. autofunction:: gymnasium.utils.step_api_compatibility.convert_to_done_step_api
```

## Runtime performance benchmark
Sometimes it is necessary to measure your environment's runtime performance, and ensure no performance regressions take place.
These tests require manual inspection of its outputs:

```{eval-rst}
.. autofunction:: gymnasium.utils.performance.benchmark_step
.. autofunction:: gymnasium.utils.performance.benchmark_vector_step
.. autofunction:: gymnasium.utils.performance.benchmark_init
.. autofunction:: gymnasium.utils.performance.benchmark_render
```

## Errors

```{eval-rst}
.. automodule:: gymnasium.error
    :members:
    :show-inheritance:
```
