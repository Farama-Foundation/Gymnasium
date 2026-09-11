"""Benchmark Box2D and Pymunk LunarLander with reproducible evidence.

Runs are explicitly labeled as smoke, pilot, or acceptance evidence. Acceptance
still depends on a sufficiently powered protocol; selecting that label does not
make a small run conclusive. Acceptance inference uses training seeds, while
episode intervals describe evaluation variability only. Normalized AUC includes
a timestep-zero evaluation and is divided by the final training timestep.

When termination and truncation coincide, termination takes precedence. Box2D
cannot publicly distinguish simultaneous hull collision and viewport exit, so
that diagnostic subtype is ``ambiguous_failure``; cross-engine summaries combine
all failure subtypes. Both engines are constructed through their registered
environment IDs so equivalent ``TimeLimit(1000)`` wrappers govern every run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import tempfile
import traceback
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pymunk_lunar_lander_terrain import physics_diagnostics  # noqa: E402

BENCHMARK_ENVIRONMENT_POLICY = {
    "box2d": {
        "id": "LunarLander-v3",
        "continuous_id": "LunarLanderContinuous-v3",
        "wrapper_chain": ["TimeLimit", "OrderEnforcing"],
        "max_episode_steps": 1_000,
    },
    "pymunk": {
        "id": "LunarLander-v4",
        "continuous_id": "LunarLanderContinuous-v4",
        "wrapper_chain": ["TimeLimit", "OrderEnforcing"],
        "max_episode_steps": 1_000,
    },
}

ACTION_MODE_BY_ALGORITHM = {
    "ppo": "discrete",
    "dqn": "discrete",
    "sac": "continuous",
}


def parse_auto_or_float(value: str) -> str | float:
    """Parse an SB3 entropy setting expressed as a number or ``auto``."""
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a number or 'auto'") from error


def parse_entropy_coefficient(value: str) -> str | float:
    """Parse SAC's fixed or automatically learned entropy coefficient."""
    if value == "auto" or value.startswith("auto_"):
        return value
    return parse_auto_or_float(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Run a reproducible LunarLander cross-engine benchmark."
    )
    parser.add_argument(
        "--algorithm", choices=list(ACTION_MODE_BY_ALGORITHM), default="ppo"
    )
    parser.add_argument(
        "--run-type",
        choices=["smoke", "pilot", "acceptance"],
        default="smoke",
        help="Label the evidentiary scope of the run; it does not alter training.",
    )
    parser.add_argument("--train-steps", type=int, default=1_000)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Fixed episode seeds shared by every engine and training seed.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=["box2d", "pymunk"],
        default=["box2d", "pymunk"],
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--episode-output-csv", type=Path, default=None)
    parser.add_argument("--summary-output-csv", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume at completed engine/seed boundaries.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--final-window-checkpoints", type=int, default=3)
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
    parser.add_argument("--sac-learning-rate", type=float, default=3e-4)
    parser.add_argument("--sac-buffer-size", type=int, default=1_000_000)
    parser.add_argument("--sac-learning-starts", type=int, default=100)
    parser.add_argument("--sac-batch-size", type=int, default=256)
    parser.add_argument("--sac-tau", type=float, default=0.005)
    parser.add_argument("--sac-gamma", type=float, default=0.99)
    parser.add_argument("--sac-train-freq", type=int, default=1)
    parser.add_argument("--sac-gradient-steps", type=int, default=1)
    parser.add_argument(
        "--sac-ent-coef", type=parse_entropy_coefficient, default="auto"
    )
    parser.add_argument("--sac-target-update-interval", type=int, default=1)
    parser.add_argument(
        "--sac-target-entropy", type=parse_auto_or_float, default="auto"
    )
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
    if args.summary_output_csv is None:
        args.summary_output_csv = args.output_csv.with_name(
            f"{args.output_csv.stem}_summary{args.output_csv.suffix}"
        )
    if args.manifest_json is None:
        args.manifest_json = args.output_csv.with_name(
            f"{args.output_csv.stem}_manifest.json"
        )
    if args.evaluation_seeds is None:
        args.evaluation_seeds = list(range(10_000, 10_000 + args.eval_episodes))
    else:
        args.eval_episodes = len(args.evaluation_seeds)
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        parser.error("--evaluation-seeds must not contain duplicates")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.final_window_checkpoints <= 0:
        parser.error("--final-window-checkpoints must be positive")
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


def environment_id(engine: str, algorithm: str) -> str:
    """Return the registered environment ID selected by an algorithm."""
    key = (
        "continuous_id" if ACTION_MODE_BY_ALGORITHM[algorithm] == "continuous" else "id"
    )
    return str(BENCHMARK_ENVIRONMENT_POLICY[engine][key])


def make_box2d_env(render_mode: str | None = None, algorithm: str = "ppo"):
    """Create the registered, time-limited Box2D benchmark environment."""
    gym = importlib.import_module("gymnasium")
    return gym.make(
        environment_id("box2d", algorithm),
        disable_env_checker=True,
        render_mode=render_mode,
    )


def make_pymunk_env(render_mode: str | None = None, algorithm: str = "ppo"):
    """Create the registered, time-limited Pymunk benchmark environment."""
    gym = importlib.import_module("gymnasium")
    return gym.make(
        environment_id("pymunk", algorithm),
        disable_env_checker=True,
        render_mode=render_mode,
    )


@dataclass
class EvaluationResult:
    """Aggregate and episode-level diagnostics for one policy evaluation."""

    mean_return: float
    median_return: float
    evaluation_return_ci_low: float
    evaluation_return_ci_high: float
    return_std: float
    return_threshold_success_rate: float
    environment_success_rate: float
    landing_rate: float
    crash_rate: float
    viewport_exit_rate: float
    ambiguous_failure_rate: float
    failure_rate: float
    time_limit_rate: float
    mean_episode_length: float
    median_episode_length: float
    evaluation_episode_length_ci_low: float
    evaluation_episode_length_ci_high: float
    termination_counts: str
    action_counts: str
    early_action_counts: str
    middle_action_counts: str
    late_action_counts: str
    continuous_action_diagnostics: str
    episodes: list[dict[str, float | int | bool | str | None]]
    settle_steps: list[dict[str, float | int | str]]


def bootstrap_confidence_interval(
    values: Sequence[float], samples: int, seed: int = 0
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap confidence interval."""
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = np.mean(array[indices], axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def classify_episode(
    observation: np.ndarray,
    terminal_reward: float,
    terminated: bool,
    truncated: bool,
) -> str:
    """Classify an episode using only engine-neutral Gymnasium outputs."""
    if terminated:
        if terminal_reward > 0.0:
            return "landing"
        if abs(float(observation[0])) >= 1.0:
            # Public outputs cannot distinguish a viewport exit from a collision
            # occurring during the same terminal transition.
            return "ambiguous_failure"
        return "crash"
    if truncated:
        return "time_limit"
    return "unknown"


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
        action: int | np.ndarray,
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
            main_power, side_power = engine_powers(action)
            self.main_engine_penalty -= 0.30 * main_power
            self.side_engine_penalty -= 0.03 * side_power
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


def engine_powers(action: int | np.ndarray) -> tuple[float, float]:
    """Reconstruct public LunarLander engine powers from an action."""
    if np.isscalar(action):
        action_int = int(action)
        return float(action_int == 2), float(action_int in (1, 3))

    continuous_action = np.asarray(action, dtype=np.float64).reshape(-1)
    if continuous_action.shape != (2,):
        raise ValueError("Continuous LunarLander actions must have shape (2,)")
    main_power = (
        float((np.clip(continuous_action[0], 0.0, 1.0) + 1.0) * 0.5)
        if continuous_action[0] > 0.0
        else 0.0
    )
    side_power = (
        float(np.clip(abs(continuous_action[1]), 0.5, 1.0))
        if abs(continuous_action[1]) > 0.5
        else 0.0
    )
    return main_power, side_power


CONTINUOUS_ACTION_DIAGNOSTIC_KEYS = (
    "main_engine_activation_count",
    "mean_active_main_power",
    "left_side_engine_activation_count",
    "right_side_engine_activation_count",
    "mean_active_side_power",
    "action_0_mean",
    "action_0_std",
    "action_1_mean",
    "action_1_std",
    "boundary_saturation_frequency",
)


def continuous_action_diagnostics(
    actions: Sequence[np.ndarray],
) -> dict[str, float | int]:
    """Summarize continuous controls using a stable, engine-neutral schema."""
    if not actions:
        return {key: 0 for key in CONTINUOUS_ACTION_DIAGNOSTIC_KEYS}
    array = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
    main_powers = [engine_powers(action)[0] for action in array]
    side_powers = [engine_powers(action)[1] for action in array]
    active_main = [power for power in main_powers if power > 0.0]
    active_side = [power for power in side_powers if power > 0.0]
    return {
        "main_engine_activation_count": len(active_main),
        "mean_active_main_power": float(np.mean(active_main)) if active_main else 0.0,
        "left_side_engine_activation_count": int(np.sum(array[:, 1] < -0.5)),
        "right_side_engine_activation_count": int(np.sum(array[:, 1] > 0.5)),
        "mean_active_side_power": float(np.mean(active_side)) if active_side else 0.0,
        "action_0_mean": float(np.mean(array[:, 0])),
        "action_0_std": float(np.std(array[:, 0])),
        "action_1_mean": float(np.mean(array[:, 1])),
        "action_1_std": float(np.std(array[:, 1])),
        # This measures actor outputs at either declared action-space boundary.
        "boundary_saturation_frequency": float(
            np.mean(np.any(np.abs(array) >= 1.0, axis=1))
        ),
    }


def action_for_environment(action: Any, action_mode: str) -> int | np.ndarray:
    """Normalize an SB3 prediction for the selected LunarLander action space."""
    if action_mode == "discrete":
        return int(action)
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.shape != (2,):
        raise ValueError("SAC must predict a continuous action with shape (2,)")
    return action_array


def evaluate_policy(
    model,
    make_env: Callable[..., Any],
    seed: int | None,
    episodes: int | None,
    success_return_threshold: float,
    post_landing_settle_steps: int = 0,
    evaluation_seeds: Sequence[int] | None = None,
    bootstrap_samples: int = 2_000,
    engine: str | None = None,
    action_mode: str = "discrete",
) -> EvaluationResult:
    """Run deterministic episodes using one shared external seed list."""
    if evaluation_seeds is None:
        if seed is None or episodes is None:
            raise ValueError("seed and episodes are required without evaluation_seeds")
        evaluation_seeds = list(range(seed, seed + episodes))
    else:
        evaluation_seeds = list(evaluation_seeds)
    if not evaluation_seeds:
        raise ValueError("evaluation_seeds must not be empty")
    returns = []
    lengths = []
    return_threshold_successes = []
    episode_logs = []
    settle_logs = []
    termination_reasons = []
    action_counts = Counter()
    early_action_counts = Counter()
    middle_action_counts = Counter()
    late_action_counts = Counter()
    all_continuous_actions: list[np.ndarray] = []

    for episode, evaluation_seed in enumerate(evaluation_seeds):
        env = make_env()
        observation, _ = env.reset(seed=evaluation_seed)
        initial_previous_shaping = (
            RewardDecomposition.shaping(observation)
            if getattr(env.unwrapped, "prev_shaping", None) is not None
            else None
        )
        reward_decomposition = RewardDecomposition(initial_previous_shaping)
        episode_return = 0.0
        episode_length = 0
        episode_actions: list[int | np.ndarray] = []
        recent_physics_diagnostics = deque(maxlen=100)

        while True:
            action, _ = model.predict(observation, deterministic=True)
            env_action = action_for_environment(action, action_mode)
            if action_mode == "discrete":
                action_counts[int(env_action)] += 1
            else:
                all_continuous_actions.append(np.asarray(env_action).copy())
            episode_actions.append(env_action)
            observation, reward, terminated, truncated, _ = env.step(env_action)
            unwrapped_env = env.unwrapped
            physics = getattr(unwrapped_env, "_physics", None)
            if physics is not None and action_mode == "discrete":
                diagnostic_function = getattr(
                    physics, "physics_diagnostics", physics_diagnostics
                )
                physics_row = (
                    diagnostic_function(int(env_action))
                    if diagnostic_function is not physics_diagnostics
                    else diagnostic_function(physics, int(env_action))
                )
                physics_row["episode_step"] = episode_length + 1
                physics_row["stable_condition_counter"] = int(
                    getattr(unwrapped_env, "stable_landing_steps", 0)
                )
                recent_physics_diagnostics.append(physics_row)
            reward_decomposition.add_step(
                observation, env_action, float(reward), terminated
            )
            episode_return += float(reward)
            episode_length += 1

            if terminated or truncated:
                break

        n_actions = len(episode_actions)

        if n_actions > 0 and action_mode == "discrete":
            early_end = max(1, n_actions // 4)
            late_start = max(early_end, (3 * n_actions) // 4)

            early_action_counts.update(episode_actions[:early_end])
            middle_action_counts.update(episode_actions[early_end:late_start])
            late_action_counts.update(episode_actions[late_start:])

        returns.append(episode_return)
        lengths.append(episode_length)
        return_threshold_success = episode_return >= success_return_threshold
        return_threshold_successes.append(float(return_threshold_success))
        outcome = classify_episode(
            observation,
            terminal_reward=float(reward),
            terminated=terminated,
            truncated=truncated,
        )
        terminal_observation = np.asarray(observation).copy()
        termination_reasons.append(outcome)
        episode_success = outcome == "landing"
        if episode_success and post_landing_settle_steps > 0:
            settle_logs.append(
                make_settle_row(
                    observation,
                    evaluation_seed=evaluation_seed,
                    episode=episode,
                    settle_step=0,
                    termination_reason=outcome,
                )
            )
            for settle_step in range(1, post_landing_settle_steps + 1):
                settle_action = (
                    0 if action_mode == "discrete" else np.zeros(2, dtype=np.float32)
                )
                (
                    observation,
                    settle_reward,
                    settle_terminated,
                    settle_truncated,
                    _,
                ) = env.step(settle_action)
                settle_reason = classify_episode(
                    observation,
                    terminal_reward=float(settle_reward),
                    terminated=settle_terminated,
                    truncated=settle_truncated,
                )
                settle_logs.append(
                    make_settle_row(
                        observation,
                        evaluation_seed=evaluation_seed,
                        episode=episode,
                        settle_step=settle_step,
                        termination_reason=settle_reason,
                    )
                )
        episode_logs.append(
            {
                "episode": episode,
                "evaluation_seed": evaluation_seed,
                "return": episode_return,
                "length": episode_length,
                "return_threshold_success": return_threshold_success,
                "environment_success": episode_success,
                "termination_reason": outcome,
                "outcome": outcome,
                "action_counts": (
                    str(dict(Counter(episode_actions)))
                    if action_mode == "discrete"
                    else "{}"
                ),
                "continuous_action_diagnostics": (
                    json.dumps(
                        continuous_action_diagnostics(
                            [np.asarray(action) for action in episode_actions]
                        ),
                        sort_keys=True,
                    )
                    if action_mode == "continuous"
                    else "{}"
                ),
                "final_observation": json.dumps(terminal_observation.tolist()),
                "time_limit_final_100": json.dumps(
                    list(recent_physics_diagnostics) if outcome == "time_limit" else []
                ),
                **reward_decomposition.as_dict(),
            }
        )
        env.close()
    termination_counts = Counter(termination_reasons)
    return_ci_low, return_ci_high = bootstrap_confidence_interval(
        returns, bootstrap_samples
    )
    length_ci_low, length_ci_high = bootstrap_confidence_interval(
        lengths, bootstrap_samples
    )
    outcome_rates = {
        name: termination_counts[name] / len(evaluation_seeds)
        for name in (
            "landing",
            "crash",
            "viewport_exit",
            "ambiguous_failure",
            "time_limit",
        )
    }
    return EvaluationResult(
        mean_return=float(np.mean(returns)),
        median_return=float(np.median(returns)),
        evaluation_return_ci_low=return_ci_low,
        evaluation_return_ci_high=return_ci_high,
        return_std=float(np.std(returns)),
        return_threshold_success_rate=float(np.mean(return_threshold_successes)),
        environment_success_rate=outcome_rates["landing"],
        landing_rate=outcome_rates["landing"],
        crash_rate=outcome_rates["crash"],
        viewport_exit_rate=outcome_rates["viewport_exit"],
        ambiguous_failure_rate=outcome_rates["ambiguous_failure"],
        failure_rate=sum(
            outcome_rates[name]
            for name in ("crash", "viewport_exit", "ambiguous_failure")
        ),
        time_limit_rate=outcome_rates["time_limit"],
        mean_episode_length=float(np.mean(lengths)),
        median_episode_length=float(np.median(lengths)),
        evaluation_episode_length_ci_low=length_ci_low,
        evaluation_episode_length_ci_high=length_ci_high,
        termination_counts=str(dict(termination_counts)),
        action_counts=str(dict(action_counts)),
        early_action_counts=str(dict(early_action_counts)),
        middle_action_counts=str(dict(middle_action_counts)),
        late_action_counts=str(dict(late_action_counts)),
        continuous_action_diagnostics=(
            json.dumps(
                continuous_action_diagnostics(all_continuous_actions), sort_keys=True
            )
            if action_mode == "continuous"
            else "{}"
        ),
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
    validate_algorithm_action_space(args.algorithm, env.action_space)
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

    if args.algorithm == "sac":
        from stable_baselines3 import SAC

        return SAC(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=args.sac_learning_rate,
            buffer_size=args.sac_buffer_size,
            learning_starts=args.sac_learning_starts,
            batch_size=args.sac_batch_size,
            tau=args.sac_tau,
            gamma=args.sac_gamma,
            train_freq=args.sac_train_freq,
            gradient_steps=args.sac_gradient_steps,
            ent_coef=args.sac_ent_coef,
            target_update_interval=args.sac_target_update_interval,
            target_entropy=args.sac_target_entropy,
            action_noise=None,
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            n_steps=1,
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            policy_kwargs=None,
            verbose=0,
        )

    raise ValueError(f"Unsupported algorithm: {args.algorithm}")


def validate_algorithm_action_space(algorithm: str, action_space: Any) -> None:
    """Reject an algorithm paired with the wrong LunarLander action-space type."""
    gym = importlib.import_module("gymnasium")
    expected_mode = ACTION_MODE_BY_ALGORITHM[algorithm]
    valid = (
        isinstance(action_space, gym.spaces.Discrete)
        if expected_mode == "discrete"
        else isinstance(action_space, gym.spaces.Box) and action_space.shape == (2,)
    )
    if not valid:
        raise ValueError(
            f"{algorithm} requires the {expected_mode} LunarLander action space"
        )


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
    all_episode_rows: list[dict[str, float | int | str]] = []
    all_settle_rows: list[dict[str, float | int | str]] = []
    all_training_rows: list[dict[str, float | int | str]] = []
    evaluation_points = [
        0,
        *list(range(args.eval_freq, args.train_steps + 1, args.eval_freq)),
    ]
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

        def _on_training_start(self) -> None:
            self._evaluate(0)
            self.next_evaluation_index = 1

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
            if args.algorithm != "ppo":
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
                "run_type": args.run_type,
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
            all_training_rows.append(row)
            if training_diagnostics_csv is not None:
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
                seed=None,
                episodes=None,
                success_return_threshold=args.success_return_threshold,
                post_landing_settle_steps=collect_settle_steps,
                evaluation_seeds=args.evaluation_seeds,
                bootstrap_samples=args.bootstrap_samples,
                engine=engine,
                action_mode=ACTION_MODE_BY_ALGORITHM[args.algorithm],
            )
            row = {
                "algorithm": args.algorithm,
                "run_type": args.run_type,
                "engine": engine,
                "seed": seed,
                "timestep": target_steps,
                "mean_return": result.mean_return,
                "median_return": result.median_return,
                "evaluation_return_ci_low": result.evaluation_return_ci_low,
                "evaluation_return_ci_high": result.evaluation_return_ci_high,
                "return_std": result.return_std,
                # Retain the old column as an alias for downstream readers.
                "success_rate": result.return_threshold_success_rate,
                "return_threshold_success_rate": (result.return_threshold_success_rate),
                "environment_success_rate": result.environment_success_rate,
                "landing_rate": result.landing_rate,
                "crash_rate": result.crash_rate,
                "viewport_exit_rate": result.viewport_exit_rate,
                "ambiguous_failure_rate": result.ambiguous_failure_rate,
                "failure_rate": result.failure_rate,
                "time_limit_rate": result.time_limit_rate,
                "mean_episode_length": result.mean_episode_length,
                "median_episode_length": result.median_episode_length,
                "evaluation_episode_length_ci_low": (
                    result.evaluation_episode_length_ci_low
                ),
                "evaluation_episode_length_ci_high": (
                    result.evaluation_episode_length_ci_high
                ),
                "termination_counts": result.termination_counts,
                "action_counts": result.action_counts,
                "early_action_counts": result.early_action_counts,
                "middle_action_counts": result.middle_action_counts,
                "late_action_counts": result.late_action_counts,
                "continuous_action_diagnostics": result.continuous_action_diagnostics,
            }
            rows.append(row)
            if output_csv is not None:
                append_csv_row(row, output_csv)
            episode_rows = [
                {
                    "algorithm": args.algorithm,
                    "run_type": args.run_type,
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
                    "continuous_action_diagnostics": episode[
                        "continuous_action_diagnostics"
                    ],
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
            all_episode_rows.extend(episode_rows)
            if settle_output_csv is not None and result.settle_steps:
                settle_rows = [
                    {
                        "run_type": args.run_type,
                        "engine": engine,
                        "training_seed": seed,
                        "training_timestep": target_steps,
                        **settle_step,
                    }
                    for settle_step in result.settle_steps
                ]
                append_settle_csv_rows(settle_rows, settle_output_csv)
                all_settle_rows.extend(settle_rows)
            elif result.settle_steps:
                all_settle_rows.extend(
                    {
                        "run_type": args.run_type,
                        "engine": engine,
                        "training_seed": seed,
                        "training_timestep": target_steps,
                        **settle_step,
                    }
                    for settle_step in result.settle_steps
                )
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
                    f"continuous_action_diagnostics="
                    f"{episode['continuous_action_diagnostics']} "
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
    model._benchmark_episode_rows = all_episode_rows
    model._benchmark_settle_rows = all_settle_rows
    model._benchmark_training_rows = all_training_rows
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
    "run_type",
    "engine",
    "seed",
    "timestep",
    "mean_return",
    "median_return",
    "evaluation_return_ci_low",
    "evaluation_return_ci_high",
    "return_std",
    "success_rate",
    "return_threshold_success_rate",
    "environment_success_rate",
    "landing_rate",
    "crash_rate",
    "viewport_exit_rate",
    "ambiguous_failure_rate",
    "failure_rate",
    "time_limit_rate",
    "mean_episode_length",
    "median_episode_length",
    "evaluation_episode_length_ci_low",
    "evaluation_episode_length_ci_high",
    "termination_counts",
    "action_counts",
    "early_action_counts",
    "middle_action_counts",
    "late_action_counts",
    "continuous_action_diagnostics",
]

EPISODE_CSV_FIELDNAMES = [
    "algorithm",
    "run_type",
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
    "continuous_action_diagnostics",
    "time_limit_final_100",
    "position_velocity_shaping",
    "angle_shaping",
    "leg_contact_shaping",
    "main_engine_penalty",
    "side_engine_penalty",
    "terminal_reward_override",
]

SETTLE_CSV_FIELDNAMES = [
    "run_type",
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
    "run_type",
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
    """Atomically write evaluation rows to CSV."""
    atomic_write_csv(rows, output_csv, CSV_FIELDNAMES)


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
    """Atomically write episode diagnostics."""
    atomic_write_csv(rows, output_csv, EPISODE_CSV_FIELDNAMES)


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


SUMMARY_METRICS = (
    "mean_return",
    "landing_rate",
    "failure_rate",
    "crash_rate",
    "viewport_exit_rate",
    "time_limit_rate",
    "mean_episode_length",
)

SUMMARY_CSV_FIELDNAMES = [
    "algorithm",
    "run_type",
    "comparison_type",
    "engine",
    "training_seeds",
    "evaluation_seeds",
    "final_window_start",
    "final_window_end",
    "best_window_start",
    "best_window_end",
    "best_window_interpretation",
    *[
        f"{window}_{stat}_{metric}"
        for window in ("final", "best")
        for metric in SUMMARY_METRICS
        for stat in ("mean", "median", "ci_low", "ci_high", "per_seed")
    ],
    "normalized_auc_mean",
    "normalized_auc_median",
    "normalized_auc_ci_low",
    "normalized_auc_ci_high",
    "normalized_auc_per_seed",
]


def normalized_learning_curve_area(
    timesteps: Sequence[float], values: Sequence[float]
) -> float:
    """Return time-weighted mean over evaluations spanning step zero to final."""
    x = np.asarray(timesteps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if (
        len(x) < 2
        or len(x) != len(y)
        or x[0] != 0
        or x[-1] <= 0
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError("evaluations must be strictly ordered from step zero")
    return float(np.trapezoid(y, x) / x[-1])


def _seed_statistics(values: dict[int, float], samples: int) -> dict[str, Any]:
    """Summarize one value per independent training seed."""
    seed_values = list(values.values())
    low, high = bootstrap_confidence_interval(seed_values, samples)
    return {
        "mean": float(np.mean(seed_values)),
        "median": float(np.median(seed_values)),
        "ci_low": low,
        "ci_high": high,
        "per_seed": json.dumps(values),
    }


def summarize_results(
    rows: Sequence[dict[str, float | int | str]],
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    """Summarize final, best, and whole-curve performance by engine."""
    summaries = []
    for engine in sorted({str(row["engine"]) for row in rows}):
        engine_rows = [row for row in rows if row["engine"] == engine]
        timesteps = sorted({int(row["timestep"]) for row in engine_rows})
        window_size = min(args.final_window_checkpoints, len(timesteps))
        windows = [
            timesteps[index : index + window_size]
            for index in range(len(timesteps) - window_size + 1)
        ]
        final_window = windows[-1]
        best_window = max(
            windows,
            key=lambda window: np.mean(
                [
                    float(row["mean_return"])
                    for row in engine_rows
                    if int(row["timestep"]) in window
                ]
            ),
        )
        summary: dict[str, float | int | str] = {
            "algorithm": args.algorithm,
            "run_type": args.run_type,
            "comparison_type": "engine",
            "engine": engine,
            "training_seeds": json.dumps(args.seeds),
            "evaluation_seeds": json.dumps(args.evaluation_seeds),
            "final_window_start": final_window[0],
            "final_window_end": final_window[-1],
            "best_window_start": best_window[0],
            "best_window_end": best_window[-1],
            "best_window_interpretation": "descriptive; selection-biased",
        }
        for window_name, window in (("final", final_window), ("best", best_window)):
            for metric in SUMMARY_METRICS:
                per_seed = {
                    seed: float(
                        np.mean(
                            [
                                float(row[metric])
                                for row in engine_rows
                                if int(row["seed"]) == seed
                                and int(row["timestep"]) in window
                            ]
                        )
                    )
                    for seed in args.seeds
                }
                statistics = _seed_statistics(per_seed, args.bootstrap_samples)
                for statistic, value in statistics.items():
                    summary[f"{window_name}_{statistic}_{metric}"] = value

        area_per_seed = {}
        for seed in args.seeds:
            seed_rows = sorted(
                (row for row in engine_rows if int(row["seed"]) == seed),
                key=lambda row: int(row["timestep"]),
            )
            x = np.asarray([float(row["timestep"]) for row in seed_rows])
            y = np.asarray([float(row["mean_return"]) for row in seed_rows])
            area_per_seed[seed] = normalized_learning_curve_area(x, y)
        for statistic, value in _seed_statistics(
            area_per_seed, args.bootstrap_samples
        ).items():
            summary[f"normalized_auc_{statistic}"] = value
        summaries.append(summary)

    if {"box2d", "pymunk"} <= {str(row["engine"]) for row in rows}:
        common_timesteps = sorted(
            set.intersection(
                *[
                    {int(row["timestep"]) for row in rows if row["engine"] == engine}
                    for engine in ("box2d", "pymunk")
                ]
            )
        )
        window_size = min(args.final_window_checkpoints, len(common_timesteps))
        final_window = common_timesteps[-window_size:]
        paired: dict[str, float | int | str] = {
            "algorithm": args.algorithm,
            "run_type": args.run_type,
            "comparison_type": "paired_box2d_minus_pymunk",
            "engine": "box2d_minus_pymunk",
            "training_seeds": json.dumps(args.seeds),
            "evaluation_seeds": json.dumps(args.evaluation_seeds),
            "final_window_start": final_window[0],
            "final_window_end": final_window[-1],
            "best_window_interpretation": "not applicable",
        }
        for metric in (
            "mean_return",
            "landing_rate",
            "failure_rate",
            "mean_episode_length",
        ):
            differences = {}
            for seed in args.seeds:
                means = {}
                for engine in ("box2d", "pymunk"):
                    means[engine] = float(
                        np.mean(
                            [
                                float(row[metric])
                                for row in rows
                                if row["engine"] == engine
                                and int(row["seed"]) == seed
                                and int(row["timestep"]) in final_window
                            ]
                        )
                    )
                differences[seed] = means["box2d"] - means["pymunk"]
            for statistic, value in _seed_statistics(
                differences, args.bootstrap_samples
            ).items():
                paired[f"final_{statistic}_{metric}"] = value
        engine_aucs = {
            str(summary["engine"]): json.loads(str(summary["normalized_auc_per_seed"]))
            for summary in summaries
        }
        auc_differences = {
            seed: float(engine_aucs["box2d"][str(seed)])
            - float(engine_aucs["pymunk"][str(seed)])
            for seed in args.seeds
        }
        for statistic, value in _seed_statistics(
            auc_differences, args.bootstrap_samples
        ).items():
            paired[f"normalized_auc_{statistic}"] = value
        summaries.append(paired)
    return summaries


def write_summary_csv(
    rows: Sequence[dict[str, float | int | str]], output_csv: Path
) -> None:
    """Atomically write run-level benchmark summaries."""
    atomic_write_csv(rows, output_csv, SUMMARY_CSV_FIELDNAMES)


def atomic_write_csv(
    rows: Sequence[dict[str, Any]], output_csv: Path, fieldnames: Sequence[str]
) -> None:
    """Replace a CSV atomically after fully writing its new contents."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", dir=output_csv.parent, delete=False
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(file.name)
    temporary.replace(output_csv)


def read_csv_validated(path: Path, fieldnames: Sequence[str]) -> list[dict[str, str]]:
    """Read a CSV after requiring its exact current schema."""
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(fieldnames):
            raise RuntimeError(f"Incompatible CSV schema in {path}")
        return list(reader)


def deduplicate_rows(
    rows: Sequence[dict[str, Any]], key_fields: Sequence[str]
) -> list[dict[str, Any]]:
    """Reject duplicate result keys rather than silently biasing summaries."""
    seen = set()
    result = []
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        if key in seen:
            raise RuntimeError(f"Duplicate result key: {key}")
        seen.add(key)
        result.append(dict(row))
    return result


def _dependency_version(distribution: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def create_run_manifest(
    args: argparse.Namespace,
    command: Sequence[str] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build a self-contained benchmark run manifest."""
    source = source_identity(args)
    started_at = (timestamp or datetime.now(timezone.utc)).isoformat()
    expected_pairs = [
        {"engine": engine, "training_seed": seed}
        for seed in args.seeds
        for engine in args.engines
    ]
    return {
        "schema_version": 5,
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "completed_at": None,
        "failure": None,
        "expected_pairs": expected_pairs,
        "completed_pairs": [],
        "current_pair": None,
        "run_type": args.run_type,
        "git_sha": source["git_sha"],
        "dirty_worktree": source["dirty_worktree"],
        "source_identity": source,
        "command": shlex.join(command or [sys.executable, *sys.argv]),
        "python": platform.python_version(),
        "dependencies": {
            name: _dependency_version(name)
            for name in (
                "gymnasium",
                "numpy",
                "pymunk",
                "stable-baselines3",
                "torch",
                "box2d",
                "box2d-py",
            )
        },
        "algorithm": args.algorithm,
        "action_mode": ACTION_MODE_BY_ALGORITHM[args.algorithm],
        "resolved_constructor": resolved_constructor_configuration(args),
        "hyperparameters": resolved_constructor_configuration(args),
        "effective_models": {},
        "training_seeds": args.seeds,
        "evaluation_seeds": args.evaluation_seeds,
        "evaluation": {
            "episodes": len(args.evaluation_seeds),
            "deterministic": True,
            "bootstrap_samples": args.bootstrap_samples,
            "final_window_checkpoints": args.final_window_checkpoints,
        },
        "environment_configuration": benchmark_environment_configuration(args),
        "run_configuration_fingerprint": run_configuration_fingerprint(args),
    }


def describe_environment(env: Any) -> dict[str, Any]:
    """Describe an environment's registered spec, wrappers, and base constructor."""
    outer = env
    wrappers = []
    time_limit = None
    current = env
    while hasattr(current, "env"):
        wrapper = {
            "class": type(current).__name__,
            "module": type(current).__module__,
        }
        if hasattr(current, "_max_episode_steps"):
            time_limit = int(current._max_episode_steps)
            wrapper["max_episode_steps"] = time_limit
        wrappers.append(wrapper)
        current = current.env

    spec = getattr(outer, "spec", None)
    constructor_arguments = {"render_mode": getattr(current, "render_mode", None)}
    for name in (
        "continuous",
        "gravity",
        "enable_wind",
        "wind_power",
        "turbulence_power",
        "solver_iterations",
    ):
        if hasattr(current, name):
            constructor_arguments[name] = getattr(current, name)
    action_space = current.action_space
    action_space_description: dict[str, Any] = {
        "class": type(action_space).__name__,
        "module": type(action_space).__module__,
        "shape": list(action_space.shape),
        "dtype": str(action_space.dtype),
    }
    if hasattr(action_space, "low"):
        action_space_description["low"] = np.asarray(action_space.low).tolist()
        action_space_description["high"] = np.asarray(action_space.high).tolist()
    if hasattr(action_space, "n"):
        action_space_description["n"] = int(action_space.n)
    return {
        "id": spec.id if spec is not None else None,
        "spec": (
            {
                "id": spec.id,
                "entry_point": str(spec.entry_point),
                "max_episode_steps": spec.max_episode_steps,
                "kwargs": dict(spec.kwargs),
            }
            if spec is not None
            else None
        ),
        "wrapper_chain": wrappers,
        "effective_max_episode_steps": time_limit,
        "base_class": type(current).__name__,
        "base_module": type(current).__module__,
        "constructor_arguments": constructor_arguments,
        "action_space": action_space_description,
        "internal_max_episode_steps": None,
    }


def benchmark_environment_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect the registered environments selected for this benchmark run."""
    factories = {
        "box2d": partial(make_box2d_env, algorithm=args.algorithm),
        "pymunk": partial(make_pymunk_env, algorithm=args.algorithm),
    }
    environments = {}
    for engine in args.engines:
        env = factories[engine]()
        try:
            environments[engine] = describe_environment(env)
        finally:
            env.close()
    return {
        "policy": "registered_lunar_lander_v1",
        "action_mode": ACTION_MODE_BY_ALGORITHM[args.algorithm],
        "engines": list(args.engines),
        "environments": environments,
    }


def resolved_constructor_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Return every constructor option explicitly resolved by this harness."""
    common = {
        "policy": "MlpPolicy",
        "seed": "training_seed",
        "verbose": 0,
        "device": "auto",
        "tensorboard_log": None,
        "stats_window_size": 100,
        "_init_setup_model": True,
    }
    if args.algorithm == "ppo":
        return {
            **common,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "n_epochs": args.n_epochs,
            "ent_coef": args.ent_coef,
            "policy_kwargs": None,
            "rollout_buffer_class": None,
            "rollout_buffer_kwargs": None,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "normalize_advantage": True,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
        }
    if args.algorithm == "dqn":
        return {
            **common,
            "learning_rate": args.dqn_learning_rate,
            "batch_size": args.dqn_batch_size,
            "buffer_size": args.dqn_buffer_size,
            "learning_starts": args.dqn_learning_starts,
            "gamma": args.dqn_gamma,
            "target_update_interval": args.dqn_target_update_interval,
            "train_freq": args.dqn_train_freq,
            "gradient_steps": args.dqn_gradient_steps,
            "exploration_fraction": args.dqn_exploration_fraction,
            "exploration_final_eps": args.dqn_exploration_final_eps,
            "policy_kwargs": {"net_arch": [256, 256]},
            "replay_buffer_class": None,
            "replay_buffer_kwargs": None,
            "optimize_memory_usage": False,
            "max_grad_norm": 10,
        }
    return {
        **common,
        "learning_rate": args.sac_learning_rate,
        "buffer_size": args.sac_buffer_size,
        "learning_starts": args.sac_learning_starts,
        "batch_size": args.sac_batch_size,
        "tau": args.sac_tau,
        "gamma": args.sac_gamma,
        "train_freq": args.sac_train_freq,
        "gradient_steps": args.sac_gradient_steps,
        "ent_coef": args.sac_ent_coef,
        "target_update_interval": args.sac_target_update_interval,
        "target_entropy": args.sac_target_entropy,
        "action_noise": None,
        "replay_buffer_class": None,
        "replay_buffer_kwargs": None,
        "optimize_memory_usage": False,
        "n_steps": 1,
        "use_sde": False,
        "sde_sample_freq": -1,
        "use_sde_at_warmup": False,
        "policy_kwargs": None,
    }


def _sha256(data: bytes) -> str:
    """Hash bytes for source and configuration identity."""
    return hashlib.sha256(data).hexdigest()


def _git_bytes(*arguments: str) -> bytes:
    """Return raw Git command output."""
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True
    ).stdout


def source_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Fingerprint tracked, staged, and relevant untracked benchmark source."""
    git_sha = _git_bytes("rev-parse", "HEAD").decode().strip()
    output_paths = {
        path.resolve() for path in output_artifact_paths(args) if path is not None
    }
    exclusions = []
    for path in output_paths:
        try:
            relative = path.relative_to(PROJECT_ROOT)
        except ValueError:
            continue
        exclusions.append(f":(exclude){relative.as_posix()}")
    tracked_diff = _git_bytes("diff", "--binary", "--", ".", *exclusions)
    staged_diff = _git_bytes("diff", "--binary", "--cached", "--", ".", *exclusions)
    untracked = []
    source_suffixes = {".py", ".toml", ".yaml", ".yml", ".md", ".lock"}
    for raw_path in (
        _git_bytes("ls-files", "--others", "--exclude-standard").decode().splitlines()
    ):
        path = (PROJECT_ROOT / raw_path).resolve()
        if path in output_paths or path.suffix.lower() not in source_suffixes:
            continue
        untracked.append({"path": raw_path, "sha256": _sha256(path.read_bytes())})
    if args.run_type == "acceptance" and (tracked_diff or staged_diff or untracked):
        raise RuntimeError(
            "Acceptance runs require clean tracked source and no relevant untracked source files"
        )
    identity = {
        "git_sha": git_sha,
        "dirty_worktree": bool(tracked_diff or staged_diff or untracked),
        "tracked_diff_sha256": _sha256(tracked_diff),
        "staged_diff_sha256": _sha256(staged_diff),
        "relevant_untracked_files": untracked,
    }
    identity["fingerprint"] = _sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    return identity


def run_configuration_fingerprint(args: argparse.Namespace) -> str:
    """Hash all settings that determine training and evaluation results."""
    configuration = {
        "algorithm": args.algorithm,
        "action_mode": ACTION_MODE_BY_ALGORITHM[args.algorithm],
        "run_type": args.run_type,
        "train_steps": args.train_steps,
        "eval_freq": args.eval_freq,
        "training_seeds": args.seeds,
        "evaluation_seeds": args.evaluation_seeds,
        "engines": args.engines,
        "environment_configuration": benchmark_environment_configuration(args),
        "success_return_threshold": args.success_return_threshold,
        "constructor": resolved_constructor_configuration(args),
        "post_landing_settle_steps": args.post_landing_settle_steps,
        "post_landing_settle_timesteps": args.post_landing_settle_timesteps,
        "bootstrap_samples": args.bootstrap_samples,
        "final_window_checkpoints": args.final_window_checkpoints,
        "checkpoint_freq": args.checkpoint_freq,
        "record_checkpoint_videos": args.record_checkpoint_videos,
        "record_videos": args.record_videos,
        "video_seeds": args.video_seeds,
        "video_episodes": args.video_episodes,
        "video_fps": args.video_fps,
        "video_max_steps": args.video_max_steps,
        "output_paths": [
            str(path.resolve()) if path is not None else None
            for path in output_artifact_paths(args)
        ],
    }
    return _sha256(json.dumps(configuration, sort_keys=True).encode())


def write_manifest(manifest: dict[str, Any], output_json: Path) -> None:
    """Atomically write a benchmark manifest as formatted JSON."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output_json.parent, delete=False) as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(output_json)


def ensure_outputs_available(paths: Sequence[Path | None], overwrite: bool) -> None:
    """Reject existing output files unless overwriting was explicitly enabled."""
    existing = [path for path in paths if path is not None and path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {names}")


def output_artifact_paths(args: argparse.Namespace) -> list[Path | None]:
    """Enumerate declared CSV, manifest, plot, checkpoint, and video artifacts."""
    paths: list[Path | None] = [
        args.output_csv,
        args.episode_output_csv,
        args.summary_output_csv,
        args.manifest_json,
        args.output_png,
        args.settle_output_csv,
        args.training_diagnostics_csv if args.algorithm == "ppo" else None,
    ]
    checkpoint_points = (
        list(range(args.checkpoint_freq, args.train_steps + 1, args.checkpoint_freq))
        if args.checkpoint_freq
        else []
    )
    if checkpoint_points and checkpoint_points[-1] != args.train_steps:
        checkpoint_points.append(args.train_steps)
    for seed in args.seeds:
        for engine in args.engines:
            for timestep in checkpoint_points:
                stem = f"{args.algorithm}_{engine}_seed_{seed}_steps_{timestep}"
                paths.append(args.checkpoint_dir / f"{stem}.zip")
                if args.record_checkpoint_videos:
                    video_seeds = args.video_seeds or [seed + 10_000]
                    paths.extend(
                        args.video_dir / f"{stem}_eval_seed_{video_seed}.mp4"
                        for video_seed in video_seeds
                    )
            if args.record_videos:
                video_seeds = args.video_seeds or args.seeds
                for video_seed in video_seeds:
                    for episode in range(args.video_episodes):
                        if len(args.seeds) > 1:
                            name = (
                                f"{args.algorithm}_{engine}_train_seed_{seed}"
                                f"_seed_{video_seed}_episode_{episode}.mp4"
                            )
                        else:
                            name = (
                                f"{args.algorithm}_{engine}_seed_{video_seed}"
                                f"_episode_{episode}.mp4"
                            )
                        paths.append(args.video_dir / name)
    return paths


def effective_model_configuration(
    model: Any, env: Any, args: argparse.Namespace | None = None
) -> dict[str, Any]:
    """Describe the effective SB3 model, policy, optimizer, and wrapped environment."""
    policy = model.policy
    activation = getattr(policy, "activation_fn", None)
    optimizer = getattr(policy, "optimizer", None)
    current = env.envs[0] if hasattr(env, "envs") else env
    environment = describe_environment(current)
    actor = getattr(model, "actor", None)
    critic = getattr(model, "critic", None)
    replay_buffer = getattr(model, "replay_buffer", None)
    actor_optimizer = getattr(actor, "optimizer", None)
    critic_optimizer = getattr(critic, "optimizer", None)
    entropy_optimizer = getattr(model, "ent_coef_optimizer", None)
    if getattr(model, "log_ent_coef", None) is not None:
        effective_ent_coef = float(model.log_ent_coef.detach().exp().item())
    elif getattr(model, "ent_coef_tensor", None) is not None:
        effective_ent_coef = float(model.ent_coef_tensor.item())
    else:
        effective_ent_coef = None
    return {
        "model_class": type(model).__name__,
        "policy_class": type(policy).__name__,
        "network_architecture": getattr(policy, "net_arch", None),
        "activation_function": (
            f"{activation.__module__}.{activation.__qualname__}" if activation else None
        ),
        "optimizer_class": type(optimizer).__name__ if optimizer is not None else None,
        "device": str(model.device),
        "n_steps": getattr(model, "n_steps", None),
        "batch_size": getattr(model, "batch_size", None),
        "gamma": getattr(model, "gamma", None),
        "learning_rate": getattr(model, "learning_rate", None),
        "train_freq": str(getattr(model, "train_freq", None)),
        "gradient_steps": getattr(model, "gradient_steps", None),
        "buffer_size": getattr(model, "buffer_size", None),
        "tau": getattr(model, "tau", None),
        "target_update_interval": getattr(model, "target_update_interval", None),
        "requested_ent_coef": getattr(model, "ent_coef", None),
        "effective_ent_coef": effective_ent_coef,
        "requested_target_entropy": (
            args.sac_target_entropy
            if args is not None and args.algorithm == "sac"
            else None
        ),
        "effective_target_entropy": getattr(model, "target_entropy", None),
        "action_noise": (
            type(model.action_noise).__name__
            if getattr(model, "action_noise", None) is not None
            else None
        ),
        "use_sde": getattr(model, "use_sde", None),
        "actor_class": type(actor).__name__ if actor is not None else None,
        "critic_class": type(critic).__name__ if critic is not None else None,
        "actor_architecture": getattr(actor, "net_arch", None),
        "critic_architecture": getattr(policy, "net_arch", None) if critic else None,
        "actor_optimizer_class": (
            type(actor_optimizer).__name__ if actor_optimizer is not None else None
        ),
        "critic_optimizer_class": (
            type(critic_optimizer).__name__ if critic_optimizer is not None else None
        ),
        "entropy_optimizer_class": (
            type(entropy_optimizer).__name__ if entropy_optimizer is not None else None
        ),
        "critic_count": len(getattr(critic, "q_networks", [])) if critic else None,
        "replay_buffer_class": (
            type(replay_buffer).__name__ if replay_buffer is not None else None
        ),
        "replay_buffer_capacity": getattr(replay_buffer, "buffer_size", None),
        "replay_buffer_handles_timeouts": getattr(
            replay_buffer, "handle_timeout_termination", None
        ),
        "environment_class": environment["base_class"],
        "wrappers": environment["wrapper_chain"],
        "time_limit_max_episode_steps": environment["effective_max_episode_steps"],
        "internal_max_episode_steps": environment["internal_max_episode_steps"],
        "constructor_arguments": environment["constructor_arguments"],
        "environment": environment,
    }


def _pair_key(engine: str, seed: int) -> tuple[str, str]:
    """Return the canonical engine/training-seed key."""
    return engine, str(seed)


def discard_incomplete_pair_artifacts(
    args: argparse.Namespace, pair: dict[str, Any] | None
) -> None:
    """Remove checkpoint artifacts belonging to an explicitly incomplete pair."""
    if pair is None or not args.checkpoint_freq:
        return
    engine = str(pair["engine"])
    seed = int(pair["training_seed"])
    points = list(
        range(args.checkpoint_freq, args.train_steps + 1, args.checkpoint_freq)
    )
    if points and points[-1] != args.train_steps:
        points.append(args.train_steps)
    for timestep in points:
        stem = f"{args.algorithm}_{engine}_seed_{seed}_steps_{timestep}"
        (args.checkpoint_dir / f"{stem}.zip").unlink(missing_ok=True)
        if args.record_checkpoint_videos:
            for video_seed in args.video_seeds or [seed + 10_000]:
                (args.video_dir / f"{stem}_eval_seed_{video_seed}.mp4").unlink(
                    missing_ok=True
                )


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
        "box2d": partial(make_box2d_env, algorithm=args.algorithm),
        "pymunk": partial(make_pymunk_env, algorithm=args.algorithm),
    }
    engines = {name: available_engines[name] for name in args.engines}

    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    settle_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    models = {}
    if args.resume and args.overwrite:
        raise RuntimeError("--resume and --overwrite are mutually exclusive")
    if args.resume:
        if not args.manifest_json.exists():
            raise RuntimeError("Cannot resume without an existing manifest")
        manifest = json.loads(args.manifest_json.read_text())
        if manifest.get("schema_version") != 5:
            raise RuntimeError(
                "Cannot resume an incompatible manifest schema; schema 4 runs "
                "cannot be migrated to the engine-neutral outcome schema 5"
            )
        if manifest.get("status") == "completed":
            raise RuntimeError("Run is already completed")
        if manifest.get(
            "run_configuration_fingerprint"
        ) != run_configuration_fingerprint(args):
            raise RuntimeError("Cannot resume with a different run configuration")
        if (
            manifest.get("source_identity", {}).get("fingerprint")
            != source_identity(args)["fingerprint"]
        ):
            raise RuntimeError("Cannot resume with a different Git source identity")
        if args.record_videos and manifest.get("completed_pairs"):
            raise RuntimeError(
                "Resume with final videos requires retained completed models"
            )
        discard_incomplete_pair_artifacts(args, manifest.get("current_pair"))
        rows = read_csv_validated(args.output_csv, CSV_FIELDNAMES)
        episode_rows = read_csv_validated(
            args.episode_output_csv, EPISODE_CSV_FIELDNAMES
        )
        if args.settle_output_csv is not None:
            settle_rows = read_csv_validated(
                args.settle_output_csv, SETTLE_CSV_FIELDNAMES
            )
        if args.training_diagnostics_csv is not None and args.algorithm == "ppo":
            training_rows = read_csv_validated(
                args.training_diagnostics_csv, TRAINING_DIAGNOSTICS_CSV_FIELDNAMES
            )
        completed = {
            _pair_key(pair["engine"], pair["training_seed"])
            for pair in manifest["completed_pairs"]
        }
        rows = [
            row for row in rows if _pair_key(row["engine"], row["seed"]) in completed
        ]
        episode_rows = [
            row
            for row in episode_rows
            if _pair_key(row["engine"], row["training_seed"]) in completed
        ]
        settle_rows = [
            row
            for row in settle_rows
            if _pair_key(row["engine"], row["training_seed"]) in completed
        ]
        training_rows = [
            row
            for row in training_rows
            if _pair_key(row["engine"], row["training_seed"]) in completed
        ]
        manifest.update(
            status="running",
            updated_at=datetime.now(timezone.utc).isoformat(),
            completed_at=None,
            failure=None,
            current_pair=None,
        )
    else:
        ensure_outputs_available(output_artifact_paths(args), args.overwrite)
        manifest = create_run_manifest(args)
        completed = set()
    write_csv(deduplicate_rows(rows, ("engine", "seed", "timestep")), args.output_csv)
    write_episode_csv(
        deduplicate_rows(
            episode_rows,
            ("engine", "training_seed", "timestep", "evaluation_seed"),
        ),
        args.episode_output_csv,
    )
    if args.settle_output_csv is not None:
        atomic_write_csv(settle_rows, args.settle_output_csv, SETTLE_CSV_FIELDNAMES)
    if args.training_diagnostics_csv is not None and args.algorithm == "ppo":
        atomic_write_csv(
            training_rows,
            args.training_diagnostics_csv,
            TRAINING_DIAGNOSTICS_CSV_FIELDNAMES,
        )
    write_manifest(manifest, args.manifest_json)
    try:
        for seed in args.seeds:
            for engine, make_env in engines.items():
                pair_key = _pair_key(engine, seed)
                if pair_key in completed:
                    continue
                pair = {"engine": engine, "training_seed": seed}
                manifest.update(
                    status="running",
                    current_pair=pair,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    failure=None,
                )
                write_manifest(manifest, args.manifest_json)
                engine_rows, model = train_and_evaluate(engine, make_env, seed, args)
                rows.extend(engine_rows)
                episode_rows.extend(model._benchmark_episode_rows)
                settle_rows.extend(model._benchmark_settle_rows)
                training_rows.extend(model._benchmark_training_rows)
                rows = deduplicate_rows(rows, ("engine", "seed", "timestep"))
                episode_rows = deduplicate_rows(
                    episode_rows,
                    ("engine", "training_seed", "timestep", "evaluation_seed"),
                )
                write_csv(rows, args.output_csv)
                write_episode_csv(episode_rows, args.episode_output_csv)
                if args.settle_output_csv is not None:
                    atomic_write_csv(
                        settle_rows, args.settle_output_csv, SETTLE_CSV_FIELDNAMES
                    )
                if (
                    args.training_diagnostics_csv is not None
                    and args.algorithm == "ppo"
                ):
                    atomic_write_csv(
                        training_rows,
                        args.training_diagnostics_csv,
                        TRAINING_DIAGNOSTICS_CSV_FIELDNAMES,
                    )
                models[(args.algorithm, engine, seed)] = model
                manifest["effective_models"][f"{engine}:{seed}"] = (
                    effective_model_configuration(model, model.get_env(), args)
                )
                manifest["completed_pairs"].append(pair)
                completed.add(pair_key)
                manifest.update(
                    current_pair=None, updated_at=datetime.now(timezone.utc).isoformat()
                )
                write_manifest(manifest, args.manifest_json)

        write_summary_csv(summarize_results(rows, args), args.summary_output_csv)
        write_plot(rows, args.output_png)
        if args.record_videos:
            record_videos(models, engines, args)
        completed_at = datetime.now(timezone.utc).isoformat()
        manifest.update(
            status="completed",
            current_pair=None,
            updated_at=completed_at,
            completed_at=completed_at,
            failure=None,
        )
        write_manifest(manifest, args.manifest_json)
    except BaseException as error:
        manifest.update(
            status="failed",
            updated_at=datetime.now(timezone.utc).isoformat(),
            failure={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        write_manifest(manifest, args.manifest_json)
        raise
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.episode_output_csv}")
    print(f"Wrote {args.summary_output_csv}")
    print(f"Wrote {args.manifest_json}")
    if args.settle_output_csv is not None:
        print(f"Wrote {args.settle_output_csv}")
    if args.training_diagnostics_csv is not None and args.algorithm == "ppo":
        print(f"Wrote {args.training_diagnostics_csv}")
    print(f"Wrote {args.output_png}")
    if args.record_videos:
        print(f"Wrote videos to {args.video_dir}")


if __name__ == "__main__":
    main()
