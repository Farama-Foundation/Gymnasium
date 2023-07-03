"""Module of wrapper classes.

Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly.
Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular. Wrappers can
also be chained to combine their effects.
Most environments that are generated via :meth:`gymnasium.make` will already be wrapped by default.

In order to wrap an environment, you must first initialize a base environment. Then you can pass this environment along
with (possibly optional) parameters to the wrapper's constructor.

    >>> import gymnasium as gym
    >>> from gymnasium.wrappers import RescaleAction
    >>> base_env = gym.make("Hopper-v4")
    >>> base_env.action_space
    Box(-1.0, 1.0, (3,), float32)
    >>> wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
    >>> wrapped_env.action_space
    Box(0.0, 1.0, (3,), float32)

You can access the environment underneath the **first** wrapper by using the :attr:`gymnasium.Wrapper.env` attribute.
As the :class:`gymnasium.Wrapper` class inherits from :class:`gymnasium.Env` then :attr:`gymnasium.Wrapper.env` can be another wrapper.

    >>> wrapped_env
    <RescaleAction<TimeLimit<OrderEnforcing<PassiveEnvChecker<HopperEnv<Hopper-v4>>>>>>
    >>> wrapped_env.env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<HopperEnv<Hopper-v4>>>>>

If you want to get to the environment underneath **all** of the layers of wrappers, you can use the
:attr:`gymnasium.Wrapper.unwrapped` attribute.
If the environment is already a bare environment, the :attr:`gymnasium.Wrapper.unwrapped` attribute will just return itself.

    >>> wrapped_env
    <RescaleAction<TimeLimit<OrderEnforcing<PassiveEnvChecker<HopperEnv<Hopper-v4>>>>>>
    >>> wrapped_env.unwrapped # doctest: +SKIP
    <gymnasium.envs.mujoco.hopper_v4.HopperEnv object at 0x7fbb5efd0490>

There are three common things you might want a wrapper to do:

- Transform actions before applying them to the base environment
- Transform observations that are returned by the base environment
- Transform rewards that are returned by the base environment

Such wrappers can be easily implemented by inheriting from :class:`gymnasium.ActionWrapper`,
:class:`gymnasium.ObservationWrapper`, or :class:`gymnasium.RewardWrapper` and implementing the respective transformation.
If you need a wrapper to do more complicated tasks, you can inherit from the :class:`gymnasium.Wrapper` class directly.

If you'd like to implement your own custom wrapper, check out `the corresponding tutorial <../../tutorials/implementing_custom_wrappers>`_.
"""
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.autoreset import AutoResetWrapper
from gymnasium.wrappers.clip_action import ClipAction
from gymnasium.wrappers.compatibility import EnvCompatibility
from gymnasium.wrappers.env_checker import PassiveEnvChecker
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.frame_stack import FrameStack, LazyFrames
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.human_rendering import HumanRendering
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
from gymnasium.wrappers.order_enforcing import OrderEnforcing
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.record_video import RecordVideo, capped_cubic_video_schedule
from gymnasium.wrappers.render_collection import RenderCollection
from gymnasium.wrappers.rescale_action import RescaleAction
from gymnasium.wrappers.resize_observation import ResizeObservation
from gymnasium.wrappers.step_api_compatibility import StepAPICompatibility
from gymnasium.wrappers.time_aware_observation import TimeAwareObservation
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.transform_reward import TransformReward
from gymnasium.wrappers.vector_list_info import VectorListInfo
