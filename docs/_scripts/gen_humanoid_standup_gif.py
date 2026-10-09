"""Render HumanoidStandup with Minari's learned SAC policy.

Install ``gymnasium[mujoco]``, ``stable-baselines3``, ``huggingface-hub`` and
``pillow``, then run this script from the repository. For headless rendering,
set ``MUJOCO_GL=egl`` before running Python.

The checkpoint is by Kallinteris Andreas for the Minari expert dataset:
https://minari.farama.org/datasets/mujoco/humanoidstandup/expert-v0/
Its inputs are joint positions and velocities, with the optional inertial,
body-velocity, actuator-force and contact-force observations disabled.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
from stable_baselines3 import SAC

import gymnasium as gym


def main():
    """Save a seeded rollout, with short pauses before and after the motion."""
    checkpoint = hf_hub_download(
        repo_id="farama-minari/HumanoidStandup-v5-SAC-expert",
        filename="humanoidstandup-v5-SAC-expert.zip",
        revision="a8a03edfc28ef8713f73b3d25445eedd4bbb776d",
    )
    model = SAC.load(checkpoint, device="cpu")
    env = gym.make(
        "HumanoidStandup-v5",
        render_mode="rgb_array",
        width=480,
        height=480,
        include_cinert_in_observation=False,
        include_cvel_in_observation=False,
        include_qfrc_actuator_in_observation=False,
        include_cfrc_ext_in_observation=False,
    )
    frames = []
    try:
        observation, _ = env.reset(seed=0)
        # Every two steps is 30 ms of simulation time, exactly representable
        # by GIF's 10 ms timing units.
        for step in range(201):
            if step % 2 == 0:
                frames.append(Image.fromarray(env.render()))
            if step == 200:
                break
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()

    output = (
        Path(__file__).resolve().parents[1]
        / "_static/videos/mujoco/humanoid_standup.gif"
    )
    durations = [30] * len(frames)
    durations[0] = durations[-1] = 500
    # A shared palette keeps the animation small without dithering the floor.
    palette = frames[0].quantize(colors=128)
    frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    main()
