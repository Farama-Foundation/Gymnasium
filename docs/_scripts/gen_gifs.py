import os
import re

import numpy as np
from PIL import Image

import gymnasium as gym
from gymnasium.envs.registration import find_highest_version, get_env_id

# how many steps to record an env for
LENGTH = 300

# envs a uniformly random policy illustrates poorly, recorded from a policy
# trained with Q-learning instead
learned_policy_env_names = [
    "CliffWalking",
]


def learn_policy(env_id, episodes=2000, alpha=0.5, gamma=0.99, seed=0):
    """Tabular Q-learning, returns the greedy action for each state."""
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(episodes):
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.5))
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))

        for _ in range(200):
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                action = int(q_table[obs].argmax())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            q_table[obs, action] += alpha * (
                reward
                + gamma * q_table[next_obs].max() * (not terminated)
                - q_table[obs, action]
            )
            obs = next_obs

            if terminated or truncated:
                break

    env.close()
    return q_table.argmax(axis=1)


exclude_env_names = [
    "GymV21Environment",
    "GymV26Environment",
    "CliffWalkingSlippery",
    "FrozenLake8x8",
    "LunarLanderContinuous",
    "BipedalWalkerHardcore",
    "phys2d/CartPole",
    "phys2d/Pendulum",
    "tabular/Blackjack",
    "tabular/CliffWalking",
]
for env_spec in gym.registry.values():
    if get_env_id(env_spec.namespace, env_spec.name, None) in exclude_env_names:
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

            policy = None
            if env_spec.name in learned_policy_env_names:
                policy = learn_policy(env_spec.id)

            # obtain and save LENGTH frames worth of steps
            frames = []
            obs, _ = env.reset()
            while len(frames) <= LENGTH:
                frames.append(Image.fromarray(env.render()))

                if policy is None:
                    action = env.action_space.sample()
                else:
                    action = int(policy[obs])

                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    frames.append(Image.fromarray(env.render()))
                    obs, _ = env.reset()

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
