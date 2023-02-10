"""Utilities of visualising an environment."""
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import Env, logger
from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled
from gymnasium.logger import deprecation


try:
    import pygame
    from pygame import Surface
    from pygame.event import Event
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        "pygame is not installed, run `pip install gymnasium[classic-control]`"
    ) from e

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except ImportError:
    logger.warn("matplotlib is not installed, run `pip install gymnasium[other]`")
    matplotlib, plt = None, None


class MissingKeysToAction(Exception):
    """Raised when the environment does not have a default ``keys_to_action`` mapping."""


class PlayableGame:
    """Wraps an environment allowing keyboard inputs to interact with the environment."""

    def __init__(
        self,
        env: Env,
        keys_to_action: Optional[Dict[Tuple[int, ...], int]] = None,
        zoom: Optional[float] = None,
    ):
        """Wraps an environment with a dictionary of keyboard buttons to action and if to zoom in on the environment.

        Args:
            env: The environment to play
            keys_to_action: The dictionary of keyboard tuples and action value
            zoom: If to zoom in on the environment render
        """
        if env.render_mode not in {"rgb_array", "rgb_array_list"}:
            raise ValueError(
                "PlayableGame wrapper works only with rgb_array and rgb_array_list render modes, "
                f"but your environment render_mode = {env.render_mode}."
            )

        self.env = env
        self.relevant_keys = self._get_relevant_keys(keys_to_action)
        # self.video_size is the size of the video that is being displayed.
        # The window size may be larger, in that case we will add black bars
        self.video_size = self._get_video_size(zoom)
        self.screen = pygame.display.set_mode(self.video_size, pygame.RESIZABLE)
        self.pressed_keys = []
        self.running = True

    def _get_relevant_keys(
        self, keys_to_action: Optional[Dict[Tuple[int], int]] = None
    ) -> set:
        if keys_to_action is None:
            if hasattr(self.env, "get_keys_to_action"):
                keys_to_action = self.env.get_keys_to_action()
            elif hasattr(self.env.unwrapped, "get_keys_to_action"):
                keys_to_action = self.env.unwrapped.get_keys_to_action()
            else:
                assert self.env.spec is not None
                raise MissingKeysToAction(
                    f"{self.env.spec.id} does not have explicit key to action mapping, "
                    "please specify one manually"
                )
        assert isinstance(keys_to_action, dict)
        relevant_keys = set(sum((list(k) for k in keys_to_action.keys()), []))
        return relevant_keys

    def _get_video_size(self, zoom: Optional[float] = None) -> Tuple[int, int]:
        rendered = self.env.render()
        if isinstance(rendered, List):
            rendered = rendered[-1]
        assert rendered is not None and isinstance(rendered, np.ndarray)
        video_size = (rendered.shape[1], rendered.shape[0])

        if zoom is not None:
            video_size = (int(video_size[0] * zoom), int(video_size[1] * zoom))

        return video_size

    def process_event(self, event: Event):
        """Processes a PyGame event.

        In particular, this function is used to keep track of which buttons are currently pressed
        and to exit the :func:`play` function when the PyGame window is closed.

        Args:
            event: The event to process
        """
        if event.type == pygame.KEYDOWN:
            if event.key in self.relevant_keys:
                self.pressed_keys.append(event.key)
            elif event.key == pygame.K_ESCAPE:
                self.running = False
        elif event.type == pygame.KEYUP:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)
        elif event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.WINDOWRESIZED:
            # Compute the maximum video size that fits into the new window
            scale_width = event.x / self.video_size[0]
            scale_height = event.y / self.video_size[1]
            scale = min(scale_height, scale_width)
            self.video_size = (scale * self.video_size[0], scale * self.video_size[1])


def display_arr(
    screen: Surface, arr: np.ndarray, video_size: Tuple[int, int], transpose: bool
):
    """Displays a numpy array on screen.

    Args:
        screen: The screen to show the array on
        arr: The array to show
        video_size: The video size of the screen
        transpose: If to transpose the array on the screen
    """
    arr_min, arr_max = np.min(arr), np.max(arr)
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    # We might have to add black bars if surface_size is larger than video_size
    surface_size = screen.get_size()
    width_offset = (surface_size[0] - video_size[0]) / 2
    height_offset = (surface_size[1] - video_size[1]) / 2
    screen.fill((0, 0, 0))
    screen.blit(pyg_img, (width_offset, height_offset))


def play(
    env: Env,
    transpose: Optional[bool] = True,
    fps: Optional[int] = None,
    zoom: Optional[float] = None,
    callback: Optional[Callable] = None,
    keys_to_action: Optional[Dict[Union[Tuple[Union[str, int]], str], ActType]] = None,
    seed: Optional[int] = None,
    noop: ActType = 0,
):
    """Allows one to play the game using keyboard.

    Args:
        env: Environment to use for playing.
        transpose: If this is ``True``, the output of observation is transposed. Defaults to ``True``.
        fps: Maximum number of steps of the environment executed every second. If ``None`` (the default),
            ``env.metadata["render_fps""]`` (or 30, if the environment does not specify "render_fps") is used.
        zoom: Zoom the observation in, ``zoom`` amount, should be positive float
        callback: If a callback is provided, it will be executed after every step. It takes the following input:
                obs_t: observation before performing action
                obs_tp1: observation after performing action
                action: action that was executed
                rew: reward that was received
                terminated: whether the environment is terminated or not
                truncated: whether the environment is truncated or not
                info: debug info
        keys_to_action:  Mapping from keys pressed to action performed.
            Different formats are supported: Key combinations can either be expressed as a tuple of unicode code
            points of the keys, as a tuple of characters, or as a string where each character of the string represents
            one key.
            For example if pressing 'w' and space at the same time is supposed
            to trigger action number 2 then ``key_to_action`` dict could look like this:

                >>> key_to_action = {
                ...    # ...
                ...    (ord('w'), ord(' ')): 2
                ...    # ...
                ... }

            or like this:

                >>> key_to_action = {
                ...    # ...
                ...    ("w", " "): 2
                ...    # ...
                ... }

            or like this:

                >>> key_to_action = {
                ...    # ...
                ...    "w ": 2
                ...    # ...
                ... }

            If ``None``, default ``key_to_action`` mapping for that environment is used, if provided.
        seed: Random seed used when resetting the environment. If None, no seed is used.
        noop: The action used when no key input has been entered, or the entered key combination is unknown.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.utils.play import play
        >>> play(gym.make("CarRacing-v2", render_mode="rgb_array"), keys_to_action={  # doctest: +SKIP
        ...                                                "w": np.array([0, 0.7, 0]),
        ...                                                "a": np.array([-1, 0, 0]),
        ...                                                "s": np.array([0, 0, 1]),
        ...                                                "d": np.array([1, 0, 0]),
        ...                                                "wa": np.array([-1, 0.7, 0]),
        ...                                                "dw": np.array([1, 0.7, 0]),
        ...                                                "ds": np.array([1, 0, 1]),
        ...                                                "as": np.array([-1, 0, 1]),
        ...                                               }, noop=np.array([0,0,0]))

        Above code works also if the environment is wrapped, so it's particularly useful in
        verifying that the frame-level preprocessing does not render the game
        unplayable.

        If you wish to plot real time statistics as you play, you can use
        :class:`gym.utils.play.PlayPlot`. Here's a sample code for plotting the reward
        for last 150 steps.

        >>> import gymnasium as gym
        >>> from gymnasium.utils.play import PlayPlot, play
        >>> def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        ...        return [rew,]
        >>> plotter = PlayPlot(callback, 150, ["reward"])             # doctest: +SKIP
        >>> play(gym.make("CartPole-v1"), callback=plotter.callback)  # doctest: +SKIP
    """
    env.reset(seed=seed)

    if keys_to_action is None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert env.spec is not None
            raise MissingKeysToAction(
                f"{env.spec.id} does not have explicit key to action mapping, "
                "please specify one manually"
            )
    assert keys_to_action is not None

    key_code_to_action = {}
    for key_combination, action in keys_to_action.items():
        key_code = tuple(
            sorted(ord(key) if isinstance(key, str) else key for key in key_combination)
        )
        key_code_to_action[key_code] = action

    game = PlayableGame(env, key_code_to_action, zoom)

    if fps is None:
        fps = env.metadata.get("render_fps", 30)

    done, obs = True, None
    clock = pygame.time.Clock()

    while game.running:
        if done:
            done = False
            obs = env.reset(seed=seed)
        else:
            action = key_code_to_action.get(tuple(sorted(game.pressed_keys)), noop)
            prev_obs = obs
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if callback is not None:
                callback(prev_obs, obs, action, rew, terminated, truncated, info)
        if obs is not None:
            rendered = env.render()
            if isinstance(rendered, List):
                rendered = rendered[-1]
            assert rendered is not None and isinstance(rendered, np.ndarray)
            display_arr(
                game.screen, rendered, transpose=transpose, video_size=game.video_size
            )

        # process pygame events
        for event in pygame.event.get():
            game.process_event(event)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


class PlayPlot:
    """Provides a callback to create live plots of arbitrary metrics when using :func:`play`.

    This class is instantiated with a function that accepts information about a single environment transition:
        - obs_t: observation before performing action
        - obs_tp1: observation after performing action
        - action: action that was executed
        - rew: reward that was received
        - terminated: whether the environment is terminated or not
        - truncated: whether the environment is truncated or not
        - info: debug info

    It should return a list of metrics that are computed from this data.
    For instance, the function may look like this::

        >>> def compute_metrics(obs_t, obs_tp, action, reward, terminated, truncated, info):
        ...     return [reward, info["cumulative_reward"], np.linalg.norm(action)]

    :class:`PlayPlot` provides the method :meth:`callback` which will pass its arguments along to that function
    and uses the returned values to update live plots of the metrics.

    Typically, this :meth:`callback` will be used in conjunction with :func:`play` to see how the metrics evolve as you play::

        >>> plotter = PlayPlot(compute_metrics, horizon_timesteps=200,                               # doctest: +SKIP
        ...                    plot_names=["Immediate Rew.", "Cumulative Rew.", "Action Magnitude"])
        >>> play(your_env, callback=plotter.callback)                                                # doctest: +SKIP
    """

    def __init__(
        self, callback: Callable, horizon_timesteps: int, plot_names: List[str]
    ):
        """Constructor of :class:`PlayPlot`.

        The function ``callback`` that is passed to this constructor should return
        a list of metrics that is of length ``len(plot_names)``.

        Args:
            callback: Function that computes metrics from environment transitions
            horizon_timesteps: The time horizon used for the live plots
            plot_names: List of plot titles

        Raises:
            DependencyNotInstalled: If matplotlib is not installed
        """
        deprecation(
            "`PlayPlot` is marked as deprecated and will be removed in the near future."
        )
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        if plt is None:
            raise DependencyNotInstalled(
                "matplotlib is not installed, run `pip install gymnasium[other]`"
            )

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot: List[Optional[plt.Axes]] = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(
        self,
        obs_t: ObsType,
        obs_tp1: ObsType,
        action: ActType,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """The callback that calls the provided data callback and adds the data to the plots.

        Args:
            obs_t: The observation at time step t
            obs_tp1: The observation at time step t+1
            action: The action
            rew: The reward
            terminated: If the environment is terminated
            truncated: If the environment is truncated
            info: The information from the environment
        """
        points = self.data_callback(
            obs_t, obs_tp1, action, rew, terminated, truncated, info
        )
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(
                range(xmin, xmax), list(self.data[i]), c="blue"
            )
            self.ax[i].set_xlim(xmin, xmax)

        if plt is None:
            raise DependencyNotInstalled(
                "matplotlib is not installed, run `pip install gymnasium[other]`"
            )
        plt.pause(0.000001)
