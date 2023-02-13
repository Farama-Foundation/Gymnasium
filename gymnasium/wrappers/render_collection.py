"""A wrapper that adds render collection mode to an environment."""
import copy

import gymnasium as gym


class RenderCollection(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Save collection of render frames."""

    def __init__(self, env: gym.Env, pop_frames: bool = True, reset_clean: bool = True):
        """Initialize a :class:`RenderCollection` instance.

        Args:
            env: The environment that is being wrapped
            pop_frames (bool): If true, clear the collection frames after .render() is called.
            Default value is True.
            reset_clean (bool): If true, clear the collection frames when .reset() is called.
            Default value is True.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, pop_frames=pop_frames, reset_clean=reset_clean
        )
        gym.Wrapper.__init__(self, env)

        assert env.render_mode is not None
        assert not env.render_mode.endswith("_list")
        self.frame_list = []
        self.reset_clean = reset_clean
        self.pop_frames = pop_frames

        self.metadata = copy.deepcopy(self.env.metadata)
        if f"{self.env.render_mode}_list" not in self.metadata["render_modes"]:
            self.metadata["render_modes"].append(f"{self.env.render_mode}_list")

    @property
    def render_mode(self):
        """Returns the collection render_mode name."""
        return f"{self.env.render_mode}_list"

    def step(self, *args, **kwargs):
        """Perform a step in the base environment and collect a frame."""
        output = self.env.step(*args, **kwargs)
        self.frame_list.append(self.env.render())
        return output

    def reset(self, *args, **kwargs):
        """Reset the base environment, eventually clear the frame_list, and collect a frame."""
        result = self.env.reset(*args, **kwargs)

        if self.reset_clean:
            self.frame_list = []
        self.frame_list.append(self.env.render())

        return result

    def render(self):
        """Returns the collection of frames and, if pop_frames = True, clears it."""
        frames = self.frame_list
        if self.pop_frames:
            self.frame_list = []

        return frames
