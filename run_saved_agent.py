import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro

from run_ddpg import Actor


@dataclass
class Args:
    checkpoint: str
    env_id: str
    video_folder: str = "videos"
    seed: int = 1
    cuda: bool = True
    episodes: int = 1


def main() -> None:
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    os.makedirs(args.video_folder, exist_ok=True)
    env = gym.make(args.env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args.video_folder,
        episode_trigger=lambda episode_id: episode_id < args.episodes,
        name_prefix="saved-agent",
    )

    actor_env = gym.vector.SyncVectorEnv([lambda: gym.make(args.env_id)])
    actor = Actor(actor_env).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint[0])
    actor.eval()
    actor_env.close()

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        episode_return = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(observation_tensor).squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)

        print(f"episode={episode + 1}, return={episode_return:.3f}")

    env.close()
    print(f"Video saved in {args.video_folder} for {args.env_id}")


if __name__ == "__main__":
    main()