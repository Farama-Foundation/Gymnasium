"""Compare Box2D LunarLander and the experimental Pymunk prototype.

This script is for early migration discussion only. It does not register the
Pymunk prototype as a public environment and does not imply trajectory parity
between Box2D and Pymunk.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pymunk_lunar_lander_terrain import (  # noqa: E402
    ExperimentalPymunkLunarLanderEnv,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--train-steps", type=int, default=1_000)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=["box2d", "pymunk"],
        default=["box2d", "pymunk"],
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--episode-output-csv", type=Path, default=None)
    parser.add_argument("--training-diagnostics-csv", type=Path, default=None)
    parser.add_argument("--post-landing-settle-steps", type=int, default=0)
    parser.add_argument(
        "--post-landing-settle-timesteps", type=int, nargs="+", default=[]
    )
    parser.add_argument("--settle-output-csv", type=Path, default=None)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--success-return-threshold", type=float, default=200.0)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--dqn-learning-rate", type=float, default=6.3e-4)
    parser.add_argument("--dqn-batch-size", type=int, default=128)
    parser.add_argument("--dqn-buffer-size", type=int, default=50_000)
    parser.add_argument("--dqn-learning-starts", type=int, default=0)
    parser.add_argument("--dqn-gamma", type=float, default=0.99)
    parser.add_argument("--dqn-target-update-interval", type=int, default=250)
    parser.add_argument("--dqn-train-freq", type=int, default=4)
    parser.add_argument("--dqn-gradient-steps", type=int, default=-1)
    parser.add_argument("--dqn-exploration-fraction", type=float, default=0.12)
    parser.add_argument("--dqn-exploration-final-eps", type=float, default=0.1)
    parser.add_argument("--record-videos", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=Path("lunar_lander_videos"))
    parser.add_argument("--video-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-max-steps", type=int, default=500)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="Save a model every N training timesteps (0 disables checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("lunar_lander_checkpoints"),
    )
    parser.add_argument(
        "--record-checkpoint-videos",
        action="store_true",
        help="Record evaluation videos whenever a model checkpoint is saved.",
    )
    parser.add_argument(
        "--video-smoke-test",
        action="store_true",
        help="Record one short deterministic episode for each engine after training.",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.output_csv is None:
        args.output_csv = Path(f"lunar_lander_pymunk_{args.algorithm}.csv")
    if args.episode_output_csv is None:
        args.episode_output_csv = args.output_csv.with_name(
            f"{args.output_csv.stem}_episodes{args.output_csv.suffix}"
        )
    if args.post_landing_settle_steps < 0:
        parser.error("--post-landing-settle-steps must be non-negative")
    if any(timestep < 0 for timestep in args.post_landing_settle_timesteps):
        parser.error("--post-landing-settle-timesteps must be non-negative")
    if (
        args.post_landing_settle_steps > 0
        and args.post_landing_settle_timesteps
        and args.settle_output_csv is None
    ):
        args.settle_output_csv = args.output_csv.with_name(
            f"{args.output_csv.stem}_settle{args.output_csv.suffix}"
        )
    if args.output_png is None:
        args.output_png = Path(f"lunar_lander_pymunk_{args.algorithm}.png")
    if args.video_smoke_test:
        args.record_videos = True
        args.video_episodes = 1
        args.video_max_steps = min(args.video_max_steps, 200)
    if args.checkpoint_freq < 0:
        parser.error("--checkpoint-freq must be non-negative")
    return args


def make_box2d_env(render_mode: str | None = None):
    """Create the current registered Box2D LunarLander."""
    gym = importlib.import_module("gymnasium")
    return gym.make("LunarLander-v3", disable_env_checker=True, render_mode=render_mode)


def make_pymunk_env(render_mode: str | None = None):
    """Create the private experimental Pymunk LunarLander."""
    return ExperimentalPymunkLunarLanderEnv(render_mode=render_mode)


@dataclass
class EvaluationResult:
    """Aggregate and episode-level diagnostics for one policy evaluation."""

    mean_return: float
    return_std: float
    return_threshold_success_rate: float
    environment_success_rate: float
    mean_episode_length: float
    termination_counts: str
    action_counts: str
    early_action_counts: str
    middle_action_counts: str
    late_action_counts: str
    episodes: list[dict[str, float | int | bool | str | None]]
    settle_steps: list[dict[str, float | int | str]]


@dataclass
class RewardDecomposition:
    """Reconstruct LunarLander reward terms from observations and actions."""

    previous_shaping: np.ndarray | None
    position_velocity_shaping: float = 0.0
    angle_shaping: float = 0.0
    leg_contact_shaping: float = 0.0
    main_engine_penalty: float = 0.0
    side_engine_penalty: float = 0.0
    terminal_reward: float = 0.0

    @staticmethod
    def shaping(observation: np.ndarray) -> np.ndarray:
        """Return the three observation-derived shaping terms."""
        return np.array(
            [
                -100.0 * np.hypot(observation[0], observation[1])
                - 100.0 * np.hypot(observation[2], observation[3]),
                -100.0 * abs(observation[4]),
                10.0 * observation[6] + 10.0 * observation[7],
            ],
            dtype=np.float64,
        )

    def add_step(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
    ) -> None:
        """Accumulate reward components for one environment step."""
        current_shaping = self.shaping(observation)
        if terminated:
            # Both environments replace all ordinary terms on terminal steps.
            self.terminal_reward += reward
        else:
            if self.previous_shaping is not None:
                delta = current_shaping - self.previous_shaping
                self.position_velocity_shaping += float(delta[0])
                self.angle_shaping += float(delta[1])
                self.leg_contact_shaping += float(delta[2])
            self.main_engine_penalty -= 0.30 if action == 2 else 0.0
            self.side_engine_penalty -= 0.03 if action in (1, 3) else 0.0
        self.previous_shaping = current_shaping

    def as_dict(self) -> dict[str, float]:
        """Return the accumulated components keyed by diagnostic column."""
        return {
            "position_velocity_shaping": self.position_velocity_shaping,
            "angle_shaping": self.angle_shaping,
            "leg_contact_shaping": self.leg_contact_shaping,
            "main_engine_penalty": self.main_engine_penalty,
            "side_engine_penalty": self.side_engine_penalty,
            "terminal_reward": self.terminal_reward,
        }


def evaluate_policy(
    model,
    make_env: Callable[..., Any],
    seed: int,
    episodes: int,
    success_return_threshold: float,
    post_landing_settle_steps: int = 0,
) -> EvaluationResult:
    """Run deterministic evaluation episodes."""
    returns = []
    lengths = []
    return_threshold_successes = []
    environment_successes = []
    episode_logs = []
    settle_logs = []
    termination_reasons = []
    action_counts = Counter()
    early_action_counts = Counter()
    middle_action_counts = Counter()
    late_action_counts = Counter()

    for episode in range(episodes):
        env = make_env()
        observation, _ = env.reset(seed=seed + episode)
        initial_previous_shaping = (
            RewardDecomposition.shaping(observation)
            if getattr(env.unwrapped, "prev_shaping", None) is not None
            else None
        )
        reward_decomposition = RewardDecomposition(initial_previous_shaping)
        episode_return = 0.0
        episode_length = 0
        final_info = {}
        episode_actions = []
        recent_physics_diagnostics = deque(maxlen=100)

        while True:
            action, _ = model.predict(observation, deterministic=True)
            action_int = int(action)
            action_counts[action_int] += 1
            episode_actions.append(action_int)
            observation, reward, terminated, truncated, final_info = env.step(action)
            unwrapped_env = env.unwrapped
            demo = getattr(unwrapped_env, "demo", None)
            if demo is not None and hasattr(demo, "physics_diagnostics"):
                physics_row = demo.physics_diagnostics(action_int)
                physics_row["episode_step"] = episode_length + 1
                physics_row["stable_condition_counter"] = int(
                    getattr(unwrapped_env, "stable_landing_steps", 0)
                )
                recent_physics_diagnostics.append(physics_row)
            reward_decomposition.add_step(
                observation, action_int, float(reward), terminated
            )
            episode_return += float(reward)
            episode_length += 1

            if terminated or truncated:
                break

        n_actions = len(episode_actions)

        if n_actions > 0:
            early_end = max(1, n_actions // 4)
            late_start = max(early_end, (3 * n_actions) // 4)

            early_action_counts.update(episode_actions[:early_end])
            middle_action_counts.update(episode_actions[early_end:late_start])
            late_action_counts.update(episode_actions[late_start:])

        returns.append(episode_return)
        lengths.append(episode_length)
        return_threshold_success = episode_return >= success_return_threshold
        environment_success = final_info.get("is_success")
        return_threshold_successes.append(float(return_threshold_success))
        if environment_success is not None:
            environment_successes.append(float(bool(environment_success)))

        termination_reason = final_info.get("termination_reason", "unknown")
        if termination_reason in (None, "unknown"):
            if terminated:
                termination_reason = "stable_landing" if reward > 0 else "crash"
            elif truncated:
                termination_reason = "time_limit"
        termination_reasons.append(termination_reason)
        outcome = (
            "success"
            if bool(environment_success) or termination_reason == "stable_landing"
            else termination_reason
        )
        episode_success = (
            bool(environment_success)
            or (terminated and float(reward) > 0.0)
            or termination_reason == "stable_landing"
        )
        if episode_success and post_landing_settle_steps > 0:
            settle_logs.append(
                make_settle_row(
                    observation,
                    evaluation_seed=seed + episode,
                    episode=episode,
                    settle_step=0,
                    termination_reason=termination_reason,
                )
            )
            for settle_step in range(1, post_landing_settle_steps + 1):
                observation, _, settle_terminated, settle_truncated, settle_info = (
                    env.step(0)
                )
                settle_reason = settle_info.get("termination_reason")
                if settle_reason is None:
                    if settle_terminated:
                        settle_reason = "stable_landing"
                    elif settle_truncated:
                        settle_reason = "time_limit"
                    else:
                        settle_reason = "none"
                settle_logs.append(
                    make_settle_row(
                        observation,
                        evaluation_seed=seed + episode,
                        episode=episode,
                        settle_step=settle_step,
                        termination_reason=settle_reason,
                    )
                )
        episode_logs.append(
            {
                "episode": episode,
                "evaluation_seed": seed + episode,
                "return": episode_return,
                "length": episode_length,
                "return_threshold_success": return_threshold_success,
                "environment_success": (
                    bool(environment_success)
                    if environment_success is not None
                    else None
                ),
                "termination_reason": termination_reason,
                "outcome": outcome,
                "action_counts": str(dict(Counter(episode_actions))),
                "final_observation": json.dumps(np.asarray(observation).tolist()),
                "time_limit_final_100": json.dumps(
                    list(recent_physics_diagnostics)
                    if termination_reason == "time_limit"
                    else []
                ),
                **reward_decomposition.as_dict(),
            }
        )
        env.close()
    termination_counts = Counter(termination_reasons)
    return EvaluationResult(
        mean_return=float(np.mean(returns)),
        return_std=float(np.std(returns)),
        return_threshold_success_rate=float(np.mean(return_threshold_successes)),
        environment_success_rate=(
            float(np.mean(environment_successes))
            if environment_successes
            else float("nan")
        ),
        mean_episode_length=float(np.mean(lengths)),
        termination_counts=str(dict(termination_counts)),
        action_counts=str(dict(action_counts)),
        early_action_counts=str(dict(early_action_counts)),
        middle_action_counts=str(dict(middle_action_counts)),
        late_action_counts=str(dict(late_action_counts)),
        episodes=episode_logs,
        settle_steps=settle_logs,
    )


def make_settle_row(
    observation: np.ndarray,
    evaluation_seed: int,
    episode: int,
    settle_step: int,
    termination_reason: str,
) -> dict[str, float | int | str]:
    """Convert one post-landing observation to a diagnostic row."""
    return {
        "evaluation_seed": evaluation_seed,
        "episode_index": episode,
        "settle_step": settle_step,
        "x": float(observation[0]),
        "y": float(observation[1]),
        "vx": float(observation[2]),
        "vy": float(observation[3]),
        "angle": float(observation[4]),
        "angular_velocity": float(observation[5]),
        "left_leg_contact": int(observation[6]),
        "right_leg_contact": int(observation[7]),
        "termination_reason": termination_reason,
    }


def make_algorithm(args: argparse.Namespace, env: Any, seed: int):
    """Create the requested SB3 algorithm with script defaults."""
    if args.algorithm == "ppo":
        from stable_baselines3 import PPO

        return PPO(
            "MlpPolicy",
            env,
            seed=seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            n_epochs=args.n_epochs,
            ent_coef=args.ent_coef,
            verbose=0,
        )

    if args.algorithm == "dqn":
        from stable_baselines3 import DQN

        return DQN(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=args.dqn_learning_rate,
            batch_size=args.dqn_batch_size,
            buffer_size=args.dqn_buffer_size,
            learning_starts=args.dqn_learning_starts,
            gamma=args.dqn_gamma,
            target_update_interval=args.dqn_target_update_interval,
            train_freq=args.dqn_train_freq,
            gradient_steps=args.dqn_gradient_steps,
            exploration_fraction=args.dqn_exploration_fraction,
            exploration_final_eps=args.dqn_exploration_final_eps,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
        )

    raise ValueError(f"Unsupported algorithm: {args.algorithm}")


def train_and_evaluate(
    engine: str,
    make_env: Callable[..., Any],
    seed: int,
    args: argparse.Namespace,
    output_csv: Path | None = None,
    episode_output_csv: Path | None = None,
    settle_output_csv: Path | None = None,
    training_diagnostics_csv: Path | None = None,
) -> tuple[list[dict[str, float | int | str]], Any]:
    """Train the selected algorithm and collect periodic evaluation rows."""
    from stable_baselines3.common.callbacks import BaseCallback

    env = make_env()
    env.reset(seed=seed)
    env.action_space.seed(seed)
    model = make_algorithm(args, env, seed=seed)

    rows: list[dict[str, float | int | str]] = []
    evaluation_points = list(
        range(args.eval_freq, args.train_steps + 1, args.eval_freq)
    )
    if not evaluation_points or evaluation_points[-1] != args.train_steps:
        evaluation_points.append(args.train_steps)
    checkpoint_points = (
        list(range(args.checkpoint_freq, args.train_steps + 1, args.checkpoint_freq))
        if args.checkpoint_freq
        else []
    )
    if checkpoint_points and checkpoint_points[-1] != args.train_steps:
        checkpoint_points.append(args.train_steps)

    class PeriodicEvaluationCallback(BaseCallback):
        """Run deterministic evaluations while a single learn call is active."""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.next_evaluation_index = 0
            self.next_checkpoint_index = 0
            self.last_recorded_ppo_update = 0

        def _on_rollout_start(self) -> None:
            # PPO records train/* values after the preceding rollout callback ends.
            self._record_ppo_training_update()

        def _on_step(self) -> bool:
            while (
                self.next_evaluation_index < len(evaluation_points)
                and self.num_timesteps >= evaluation_points[self.next_evaluation_index]
            ):
                self._evaluate(evaluation_points[self.next_evaluation_index])
                self.next_evaluation_index += 1
            while (
                self.next_checkpoint_index < len(checkpoint_points)
                and self.num_timesteps >= checkpoint_points[self.next_checkpoint_index]
            ):
                self._save_checkpoint(checkpoint_points[self.next_checkpoint_index])
                self.next_checkpoint_index += 1
            return True

        def _on_training_end(self) -> None:
            # The last PPO update has no subsequent rollout start at which to record it.
            self._record_ppo_training_update()
            if (
                self.next_evaluation_index < len(evaluation_points)
                and evaluation_points[-1] == args.train_steps
            ):
                self._evaluate(args.train_steps)
                self.next_evaluation_index = len(evaluation_points)

            while self.next_checkpoint_index < len(checkpoint_points):
                self._save_checkpoint(checkpoint_points[self.next_checkpoint_index])
                self.next_checkpoint_index += 1

        def _record_ppo_training_update(self) -> None:
            if args.algorithm != "ppo" or training_diagnostics_csv is None:
                return

            logger_values = self.model.logger.name_to_value
            update = int(
                logger_values.get(
                    "train/n_updates", getattr(self.model, "_n_updates", 0)
                )
            )
            if (
                update <= self.last_recorded_ppo_update
                or "train/approx_kl" not in logger_values
            ):
                return

            episode_infos = list(self.model.ep_info_buffer or [])
            episode_returns = [
                float(info["r"]) for info in episode_infos if "r" in info
            ]
            episode_lengths = [
                float(info["l"]) for info in episode_infos if "l" in info
            ]
            optimizer = getattr(self.model.policy, "optimizer", None)
            learning_rate = (
                float(optimizer.param_groups[0]["lr"])
                if optimizer is not None and optimizer.param_groups
                else ""
            )
            row: dict[str, float | int | str] = {
                "engine": engine,
                "training_seed": seed,
                "timestep": self.num_timesteps,
                "n_updates": update,
                "learning_rate": learning_rate,
                "rollout_episode_return": (
                    float(np.mean(episode_returns)) if episode_returns else ""
                ),
                "rollout_episode_length": (
                    float(np.mean(episode_lengths)) if episode_lengths else ""
                ),
            }
            for csv_name, logger_name in PPO_TRAIN_LOGGER_FIELDS.items():
                value = logger_values.get(logger_name, "")
                if isinstance(value, np.generic):
                    value = value.item()
                row[csv_name] = value
            append_training_diagnostics_row(row, training_diagnostics_csv)
            self.last_recorded_ppo_update = update

        def _evaluate(self, target_steps: int) -> None:
            collect_settle_steps = (
                args.post_landing_settle_steps
                if target_steps in args.post_landing_settle_timesteps
                else 0
            )
            result = evaluate_policy(
                self.model,
                make_env,
                # Keep the evaluation episodes identical at every checkpoint.
                seed=seed + 10_000,
                episodes=args.eval_episodes,
                success_return_threshold=args.success_return_threshold,
                post_landing_settle_steps=collect_settle_steps,
            )
            row = {
                "algorithm": args.algorithm,
                "engine": engine,
                "seed": seed,
                "timestep": target_steps,
                "mean_return": result.mean_return,
                "return_std": result.return_std,
                # Retain the old column as an alias for downstream readers.
                "success_rate": result.return_threshold_success_rate,
                "return_threshold_success_rate": (result.return_threshold_success_rate),
                "environment_success_rate": result.environment_success_rate,
                "mean_episode_length": result.mean_episode_length,
                "termination_counts": result.termination_counts,
                "action_counts": result.action_counts,
                "early_action_counts": result.early_action_counts,
                "middle_action_counts": result.middle_action_counts,
                "late_action_counts": result.late_action_counts,
            }
            rows.append(row)
            if output_csv is not None:
                append_csv_row(row, output_csv)
            episode_rows = [
                {
                    "algorithm": args.algorithm,
                    "engine": engine,
                    "training_seed": seed,
                    "timestep": target_steps,
                    "evaluation_seed": episode["evaluation_seed"],
                    "episode_index": episode["episode"],
                    "episode_return": episode["return"],
                    "episode_length": episode["length"],
                    "outcome": episode["outcome"],
                    "termination_reason": episode["termination_reason"],
                    "final_observation": episode["final_observation"],
                    "action_counts": episode["action_counts"],
                    "time_limit_final_100": episode["time_limit_final_100"],
                    "position_velocity_shaping": episode["position_velocity_shaping"],
                    "angle_shaping": episode["angle_shaping"],
                    "leg_contact_shaping": episode["leg_contact_shaping"],
                    "main_engine_penalty": episode["main_engine_penalty"],
                    "side_engine_penalty": episode["side_engine_penalty"],
                    "terminal_reward_override": episode["terminal_reward"],
                }
                for episode in result.episodes
            ]
            if episode_output_csv is not None:
                append_episode_csv_rows(episode_rows, episode_output_csv)
            if settle_output_csv is not None and result.settle_steps:
                settle_rows = [
                    {
                        "engine": engine,
                        "training_seed": seed,
                        "training_timestep": target_steps,
                        **settle_step,
                    }
                    for settle_step in result.settle_steps
                ]
                append_settle_csv_rows(settle_rows, settle_output_csv)
            for episode in result.episodes:
                print(
                    "evaluation_episode "
                    f"algorithm={args.algorithm} engine={engine} train_seed={seed} "
                    f"timestep={target_steps} eval_seed={episode['evaluation_seed']} "
                    f"episode={episode['episode']} return={episode['return']:.3f} "
                    f"length={episode['length']} "
                    f"return_threshold_success={episode['return_threshold_success']} "
                    f"environment_success={episode['environment_success']} "
                    f"outcome={episode['outcome']} "
                    f"termination_reason={episode['termination_reason']} "
                    f"action_counts={episode['action_counts']} "
                    f"final_observation={episode['final_observation']} "
                    f"position_velocity_shaping="
                    f"{episode['position_velocity_shaping']:.6f} "
                    f"angle_shaping={episode['angle_shaping']:.6f} "
                    f"leg_contact_shaping={episode['leg_contact_shaping']:.6f} "
                    f"main_engine_penalty={episode['main_engine_penalty']:.6f} "
                    f"side_engine_penalty={episode['side_engine_penalty']:.6f} "
                    f"terminal_reward={episode['terminal_reward']:.6f}"
                )

        def _save_checkpoint(self, target_steps: int) -> None:
            checkpoint_stem = (
                f"{args.algorithm}_{engine}_seed_{seed}_steps_{target_steps}"
            )
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(args.checkpoint_dir / checkpoint_stem)
            if args.record_checkpoint_videos:
                video_seeds = (
                    args.video_seeds
                    if args.video_seeds is not None
                    else [seed + 10_000]
                )
                for video_seed in video_seeds:
                    video_path = (
                        args.video_dir / f"{checkpoint_stem}_eval_seed_{video_seed}.mp4"
                    )
                    record_policy_video(
                        self.model,
                        make_env,
                        video_path,
                        seed=video_seed,
                        fps=args.video_fps,
                        max_steps=args.video_max_steps,
                    )

    model.learn(total_timesteps=args.train_steps, callback=PeriodicEvaluationCallback())
    env.close()
    return rows, model


def write_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Write collected RGB frames to an MP4 file."""
    from moviepy import ImageSequenceClip

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(frames, fps=fps)
    try:
        clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio=False,
            logger=None,
        )
    finally:
        clip.close()


def record_policy_video(
    model,
    make_env: Callable[..., Any],
    output_path: Path,
    seed: int,
    fps: int,
    max_steps: int,
) -> None:
    """Record one deterministic evaluation episode."""
    env = make_env(render_mode="rgb_array")
    frames = []
    observation, _ = env.reset(seed=seed)
    frames.append(env.render())

    for _ in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break

    env.close()
    write_video(frames, output_path, fps=fps)


def record_videos(
    models: dict[tuple[str, str, int], Any],
    engines: dict[str, Callable[..., Any]],
    args: argparse.Namespace,
) -> None:
    """Record final-policy videos for selected training and evaluation seeds."""
    video_seeds = args.video_seeds if args.video_seeds is not None else args.seeds
    include_train_seed = len(args.seeds) > 1

    for (algorithm, engine, train_seed), model in models.items():
        for video_seed in video_seeds:
            for episode in range(args.video_episodes):
                episode_seed = video_seed + episode
                if include_train_seed:
                    filename = (
                        f"{algorithm}_{engine}_train_seed_{train_seed}"
                        f"_seed_{video_seed}"
                        f"_episode_{episode}.mp4"
                    )
                else:
                    filename = (
                        f"{algorithm}_{engine}_seed_{video_seed}_episode_{episode}.mp4"
                    )

                record_policy_video(
                    model,
                    engines[engine],
                    args.video_dir / filename,
                    seed=episode_seed,
                    fps=args.video_fps,
                    max_steps=args.video_max_steps,
                )


CSV_FIELDNAMES = [
    "algorithm",
    "engine",
    "seed",
    "timestep",
    "mean_return",
    "return_std",
    "success_rate",
    "return_threshold_success_rate",
    "environment_success_rate",
    "mean_episode_length",
    "termination_counts",
    "action_counts",
    "early_action_counts",
    "middle_action_counts",
    "late_action_counts",
]

EPISODE_CSV_FIELDNAMES = [
    "algorithm",
    "engine",
    "training_seed",
    "timestep",
    "evaluation_seed",
    "episode_index",
    "episode_return",
    "episode_length",
    "outcome",
    "termination_reason",
    "final_observation",
    "action_counts",
    "time_limit_final_100",
    "position_velocity_shaping",
    "angle_shaping",
    "leg_contact_shaping",
    "main_engine_penalty",
    "side_engine_penalty",
    "terminal_reward_override",
]

SETTLE_CSV_FIELDNAMES = [
    "engine",
    "training_seed",
    "training_timestep",
    "evaluation_seed",
    "episode_index",
    "settle_step",
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
    "termination_reason",
]

PPO_TRAIN_LOGGER_FIELDS = {
    "approx_kl": "train/approx_kl",
    "clip_fraction": "train/clip_fraction",
    "entropy_loss": "train/entropy_loss",
    "policy_gradient_loss": "train/policy_gradient_loss",
    "value_loss": "train/value_loss",
    "explained_variance": "train/explained_variance",
    "loss": "train/loss",
    "clip_range": "train/clip_range",
    "clip_range_vf": "train/clip_range_vf",
    "policy_std": "train/std",
}

TRAINING_DIAGNOSTICS_CSV_FIELDNAMES = [
    "engine",
    "training_seed",
    "timestep",
    "n_updates",
    "approx_kl",
    "clip_fraction",
    "entropy_loss",
    "policy_gradient_loss",
    "value_loss",
    "explained_variance",
    "learning_rate",
    "rollout_episode_return",
    "rollout_episode_length",
    "loss",
    "clip_range",
    "clip_range_vf",
    "policy_std",
]


def write_csv(rows: list[dict[str, float | int | str]], output_csv: Path) -> None:
    """Write evaluation rows to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(row: dict[str, float | int | str], output_csv: Path) -> None:
    """Append and flush one evaluation row so interrupted runs retain results."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def write_episode_csv(
    rows: list[dict[str, float | int | str]], output_csv: Path
) -> None:
    """Write episode diagnostics, replacing any previous output."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=EPISODE_CSV_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def append_episode_csv_rows(
    rows: list[dict[str, float | int | str]], output_csv: Path
) -> None:
    """Append and flush one checkpoint's episode diagnostics."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=EPISODE_CSV_FIELDNAMES, extrasaction="ignore"
        )
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        file.flush()


def write_settle_csv(
    rows: list[dict[str, float | int | str]], output_csv: Path
) -> None:
    """Write post-landing settling diagnostics, replacing previous output."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=SETTLE_CSV_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def append_settle_csv_rows(
    rows: list[dict[str, float | int | str]], output_csv: Path
) -> None:
    """Append and flush one checkpoint's post-landing settling rows."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=SETTLE_CSV_FIELDNAMES, extrasaction="ignore"
        )
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        file.flush()


def write_training_diagnostics_csv(
    rows: list[dict[str, float | int | str]], output_csv: Path
) -> None:
    """Write PPO update diagnostics, replacing any previous output."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=TRAINING_DIAGNOSTICS_CSV_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        file.flush()


def append_training_diagnostics_row(
    row: dict[str, float | int | str], output_csv: Path
) -> None:
    """Append and flush one PPO policy-update diagnostics row."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=TRAINING_DIAGNOSTICS_CSV_FIELDNAMES,
            extrasaction="ignore",
        )
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def write_plot(rows: list[dict[str, float | int | str]], output_png: Path) -> None:
    """Write a learning-curve PNG comparing engines."""
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    series = sorted({(str(row["algorithm"]), str(row["engine"])) for row in rows})

    _, axis = plt.subplots()
    for algorithm, engine in series:
        engine_rows = [
            row
            for row in rows
            if row["algorithm"] == algorithm and row["engine"] == engine
        ]
        timesteps = sorted({int(row["timestep"]) for row in engine_rows})
        means = []
        stds = []
        for timestep in timesteps:
            returns = [
                float(row["mean_return"])
                for row in engine_rows
                if int(row["timestep"]) == timestep
            ]
            means.append(float(np.mean(returns)))
            stds.append(float(np.std(returns)))

        axis.plot(timesteps, means, marker="o", label=f"{algorithm}-{engine}")
        axis.fill_between(
            timesteps,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
        )

    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Mean deterministic return")
    axis.legend()
    axis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def main() -> None:
    """Run the comparison."""
    args = parse_args()
    available_engines = {
        "box2d": make_box2d_env,
        "pymunk": make_pymunk_env,
    }
    engines = {name: available_engines[name] for name in args.engines}

    rows = []
    models = {}
    # Truncate a previous result file once, then append after every evaluation.
    write_csv([], args.output_csv)
    write_episode_csv([], args.episode_output_csv)
    if args.settle_output_csv is not None:
        write_settle_csv([], args.settle_output_csv)
    if args.training_diagnostics_csv is not None and args.algorithm == "ppo":
        write_training_diagnostics_csv([], args.training_diagnostics_csv)
    for seed in args.seeds:
        for engine, make_env in engines.items():
            engine_rows, model = train_and_evaluate(
                engine,
                make_env,
                seed,
                args,
                output_csv=args.output_csv,
                episode_output_csv=args.episode_output_csv,
                settle_output_csv=args.settle_output_csv,
                training_diagnostics_csv=args.training_diagnostics_csv,
            )
            rows.extend(engine_rows)
            models[(args.algorithm, engine, seed)] = model

    write_plot(rows, args.output_png)
    if args.record_videos:
        record_videos(models, engines, args)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.episode_output_csv}")
    if args.settle_output_csv is not None:
        print(f"Wrote {args.settle_output_csv}")
    if args.training_diagnostics_csv is not None and args.algorithm == "ppo":
        print(f"Wrote {args.training_diagnostics_csv}")
    print(f"Wrote {args.output_png}")
    if args.record_videos:
        print(f"Wrote videos to {args.video_dir}")


if __name__ == "__main__":
    main()
