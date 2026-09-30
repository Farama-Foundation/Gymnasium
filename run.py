import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register

from envgen import generate_environment
from run_ddpg import Actor


def train(env_id, seed, total_timesteps, learning_starts, exp_name):
    command = [
        sys.executable,
        "run_ddpg.py",
        "--env-id",
        env_id,
        "--seed",
        str(seed),
        "--total-timesteps",
        str(total_timesteps),
        "--learning-starts",
        str(learning_starts),
        "--exp-name",
        exp_name,
    ]
    subprocess.run(command, check=True)
    candidates = sorted(
        Path("runs").glob(f"{env_id}__{exp_name}__{seed}__*/{exp_name}.cleanrl_model"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint was produced for {env_id}")
    return str(candidates[-1])


def evaluate_random(env_id, episodes, seed):
    returns = []
    for episode in range(episodes):
        env = gym.make(env_id)
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            episode_return += float(reward)
        returns.append(episode_return)
        env.close()
    return returns


def evaluate_learned(env_id, checkpoint, episodes, seed, cuda):
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    actor_env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id)])
    actor = Actor(actor_env).to(device)
    actor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)[0])
    actor.eval()
    returns = []
    for episode in range(episodes):
        env = gym.make(env_id)
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            with torch.no_grad():
                action = actor(torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))
            action = action.squeeze(0).cpu().numpy()
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
        returns.append(episode_return)
        env.close()
    actor_env.close()
    return returns


def summarize(random_returns, learned_returns):
    random_mean = float(np.mean(random_returns))
    learned_mean = float(np.mean(learned_returns))
    return {
        "random_returns": random_returns,
        "learned_returns": learned_returns,
        "random_mean": random_mean,
        "learned_mean": learned_mean,
        "improvement": learned_mean - random_mean,
    }


def register_generated_environment(module_name, module_path):
    importlib.invalidate_caches()
    module = importlib.import_module(f"gymnasium.envs.classic_control.{module_name}")
    environment_classes = [
        item
        for _, item in inspect.getmembers(module, inspect.isclass)
        if issubclass(item, gym.Env) and item is not gym.Env and item.__module__ == module.__name__
    ]
    if len(environment_classes) != 1:
        raise RuntimeError(f"Expected one generated Env class in {module_path}")
    env_id = f"Generated{module_name.title().replace('_', '')}-v0"
    register(id=env_id, entry_point=f"gymnasium.envs.classic_control.{module_name}:{environment_classes[0].__name__}")
    return env_id


def main():
    parser = argparse.ArgumentParser(description="Run the open-ended RL training pipeline")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=25_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    reports = []
    env_id = "Pendulum-v1"
    learned_titles = [env_id]
    for iteration in range(args.iterations + 1):
        exp_name = f"pipeline_stage_{iteration}"
        checkpoint = train(env_id, args.seed + iteration, args.total_timesteps, args.learning_starts, exp_name)
        random_returns = evaluate_random(env_id, args.episodes, args.seed + iteration)
        learned_returns = evaluate_learned(env_id, checkpoint, args.episodes, args.seed + iteration, args.cuda)
        report = summarize(random_returns, learned_returns)
        report.update({"env_id": env_id, "checkpoint": checkpoint})
        reports.append(report)
        print(json.dumps(report, indent=2))

        if iteration == args.iterations:
            break
        generated = generate_environment(learned_titles, report)
        env_id = register_generated_environment(generated["module_name"], generated["module_path"])
        learned_titles.append(generated["title"])
        print(f"Generated and registered {env_id}: {generated['title']}")

    os.makedirs("runs", exist_ok=True)
    with open("runs/pipeline_results.json", "w", encoding="utf-8") as results_file:
        json.dump(reports, results_file, indent=2)
    print("Pipeline results saved to runs/pipeline_results.json")


if __name__ == "__main__":
    main()