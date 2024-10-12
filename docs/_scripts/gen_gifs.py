import os
import re

from PIL import Image

import gymnasium as gym
from gymnasium.envs.registration import find_highest_version, get_env_id


# how many steps to record an env for
LENGTH = 300


exclude_env_names = [
    "GymV21Environment",
    "GymV26Environment",
    "FrozenLake8x8",
    "LunarLanderContinuous",
    "BipedalWalkerHardcore",
]
for env_spec in gym.registry.values():
    if env_spec.name in exclude_env_names:
        continue

    highest_version = find_highest_version(env_spec.namespace, env_spec.name)
    env_id = get_env_id(env_spec.namespace, env_spec.name, highest_version)

    if env_id == env_spec.id and isinstance(env_spec.entry_point, str):
        if "gymnasium" in env_spec.entry_point or (
            "ALE" == env_spec.namespace and env_spec.kwargs["obs_type"] == "rgb"
        ):
            print(env_spec.id)
            env = gym.make(env_spec, render_mode="rgb_array").unwrapped

            # the gymnasium needs to be rgb renderable
            if "rgb_array" not in env.metadata["render_modes"]:
                continue

            # obtain and save LENGTH frames worth of steps
            frames = []
            env.reset()
            while len(frames) <= LENGTH:
                frames.append(Image.fromarray(env.render()))

                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    env.reset()

            env.close()

            # make sure video doesn't already exist
            # if not os.path.exists(os.path.join(v_path, env_name + ".gif")):
            env_module = env_spec.entry_point.split(".")[2]
            env_name = re.sub(r"(?<!^)(?=[A-Z])", "_", env_spec.name).lower()

            # render_fps = env.metadata.get("render_fps", 30)
            video_path = os.path.join(
                "..", "_static", "videos", env_module, env_name + ".gif"
            )
            frames[0].save(
                video_path,
                save_all=True,
                append_images=frames[1:],
                duration=50,  # milliseconds for the frame
                loop=0,
            )
