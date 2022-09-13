---
title: Utils
---

# Utils

## Visualization

```{eval-rst}
.. autoclass:: gymnasium.utils.play.PlayableGame
    
    .. automethod:: process_event

.. autoclass:: gymnasium.utils.play.PlayPlot
    
    .. automethod:: callback

.. autofunction:: gymnasium.utils.play.display_arr
.. autofunction:: gymnasium.utils.play.play

```

## Save Rendering Videos

```{eval-rst}
.. autofunction:: gymnasium.utils.save_video.capped_cubic_video_schedule
.. autofunction:: gymnasium.utils.save_video.save_video
```

## Old to New Step API Compatibility

```{eval-rst}
.. autofunction:: gymnasium.utils.step_api_compatibility.convert_to_terminated_truncated_step_api
.. autofunction:: gymnasium.utils.step_api_compatibility.convert_to_done_step_api
.. autofunction:: gymnasium.utils.step_api_compatibility.step_api_compatibility
```

## Seeding

```{eval-rst}
.. autofunction:: gymnasium.utils.seeding.np_random
```

## Environment Checking

### Invasive

```{eval-rst}
.. autofunction:: gymnasium.utils.env_checker.check_env
.. autofunction:: gymnasium.utils.env_checker.data_equivalence
.. autofunction:: gymnasium.utils.env_checker.check_reset_seed
.. autofunction:: gymnasium.utils.env_checker.check_reset_options
.. autofunction:: gymnasium.utils.env_checker.check_reset_return_info_deprecation
.. autofunction:: gymnasium.utils.env_checker.check_seed_deprecation
.. autofunction:: gymnasium.utils.env_checker.check_reset_return_type
.. autofunction:: gymnasium.utils.env_checker.check_space_limit
``` 

