"""Generate a FrozenLake GIF showing a successful episode."""

from pathlib import Path

from PIL import Image

import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="rgb_array")

# Down, down, right, down, right, right: a safe route to the goal.
actions = [1, 1, 2, 1, 2, 2]
frames = []

try:
    env.reset(seed=160)
    frames.append(Image.fromarray(env.render()))

    for action in actions:
        env.step(action)
        frames.append(Image.fromarray(env.render()))

    output = (
        Path(__file__).resolve().parents[1] / "_static/videos/toy_text/frozen_lake.gif"
    )

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=[500] * (len(frames) - 1) + [1500],
        loop=0,
    )
    print(f"Saved GIF to {output}")
finally:
    env.close()
