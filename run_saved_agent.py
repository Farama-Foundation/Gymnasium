import json
import os
from pathlib import Path
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro

from run_ddpg import Actor, register_saved_generated_environments


def _patch_generated_render_helpers() -> None:
    """Provide path helpers expected by older generated render implementations."""
    from gymnasium.envs.classic_control import utils

    if not hasattr(utils, "path"):
        utils.path = os.path
    if not hasattr(utils, "path_join"):
        utils.path_join = os.path.join
    if not hasattr(utils, "path_dirname"):
        utils.path_dirname = os.path.dirname


@dataclass
class Args:
    checkpoint: str | None = None
    env_id: str | None = None
    runs_dir: str = "runs"
    video_folder: str = "videos"
    seed: int = 1
    cuda: bool = True
    episodes: int = 1
    all_envs: bool = True


def _registered_environment_ids(runs_dir: str) -> set[str]:
    environment_ids = {"Pendulum-v1"}
    registry_path = Path(runs_dir) / "generated_environments.json"
    if registry_path.is_file():
        with registry_path.open(encoding="utf-8") as registry_file:
            environment_ids.update(json.load(registry_file))
    return environment_ids


def discover_agents(runs_dir: str, environment_id: str | None = None) -> list[tuple[str, Path]]:
    """Find the newest saved checkpoint for each trained environment.

    Checkpoint directories use ``<environment_id>__<run_metadata>`` names.
    Using that name as the source of truth keeps inference independent from
    the training-time registry and automatically includes every saved agent.
    """
    checkpoints: dict[str, Path] = {}
    for checkpoint in Path(runs_dir).rglob("*.cleanrl_model"):
        run_name = checkpoint.parent.name
        if "__" not in run_name:
            continue

        matched_id = run_name.split("__", 1)[0]
        if environment_id is not None and matched_id != environment_id:
            continue

        previous = checkpoints.get(matched_id)
        checkpoint_is_newer = (
            previous is None
            or checkpoint.stat().st_mtime > previous.stat().st_mtime
        )
        if checkpoint_is_newer:
            checkpoints[matched_id] = checkpoint

    return sorted(checkpoints.items())


def _checkpoint_environment_id(checkpoint: str, runs_dir: str) -> str:
    run_name = Path(checkpoint).parent.name
    if "__" in run_name:
        return run_name.split("__", 1)[0]

    for environment_id in _registered_environment_ids(runs_dir):
        if run_name.startswith(f"{environment_id}__"):
            return environment_id
    raise ValueError(
        f"Could not infer the environment for checkpoint {checkpoint}; "
        "pass --env-id explicitly."
    )


def run_agent(
    env_id: str,
    checkpoint_path: Path,
    video_folder: str,
    seed: int,
    episodes: int,
    device: torch.device,
) -> None:
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda episode_id: episode_id < episodes,
        name_prefix=f"saved-agent-{env_id}",
    )

    actor_env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id)])
    try:
        actor = Actor(actor_env).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint[0])
        actor.eval()

        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                observation_tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    action = actor(observation_tensor).squeeze(0).cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)

            print(
                f"env={env_id}, checkpoint={checkpoint_path}, "
                f"episode={episode + 1}, return={episode_return:.3f}"
            )
    finally:
        actor_env.close()
        env.close()


def main() -> None:
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    os.makedirs(args.video_folder, exist_ok=True)
    _patch_generated_render_helpers()
    register_saved_generated_environments()
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        env_id = args.env_id or _checkpoint_environment_id(args.checkpoint, args.runs_dir)
        agents = [(env_id, checkpoint_path)]
    else:
        agents = discover_agents(args.runs_dir, args.env_id)
        if args.env_id is not None and not agents:
            raise FileNotFoundError(f"No checkpoint found for environment: {args.env_id}")
        if not args.all_envs and agents:
            agents = agents[-1:]

    if not agents:
        raise FileNotFoundError(f"No saved agents found in {args.runs_dir}")

    for env_id, checkpoint_path in agents:
        run_agent(
            env_id,
            checkpoint_path,
            args.video_folder,
            args.seed,
            args.episodes,
            device,
        )

    print(f"Videos saved in {args.video_folder} for {len(agents)} environment(s)")


if __name__ == "__main__":
    main()