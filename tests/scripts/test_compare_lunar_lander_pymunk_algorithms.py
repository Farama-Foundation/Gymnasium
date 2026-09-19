import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymunk")

from scripts import compare_lunar_lander_pymunk_algorithms as compare


def _dqn_args(*extra_args):
    return compare.parse_args(
        [
            "--algorithm",
            "dqn",
            "--train-steps",
            "4",
            "--eval-freq",
            "4",
            "--eval-episodes",
            "1",
            "--dqn-batch-size",
            "1",
            "--dqn-buffer-size",
            "16",
            "--dqn-target-update-interval",
            "1",
            *extra_args,
        ]
    )


def _sac_args(*extra_args):
    return compare.parse_args(
        [
            "--algorithm",
            "sac",
            "--train-steps",
            "2",
            "--eval-freq",
            "2",
            "--eval-episodes",
            "1",
            "--sac-batch-size",
            "2",
            "--sac-buffer-size",
            "16",
            "--sac-learning-starts",
            "0",
            *extra_args,
        ]
    )


def test_parse_args_defaults_to_ppo():
    args = compare.parse_args([])

    assert args.algorithm == "ppo"
    assert args.run_type == "smoke"
    assert args.evaluation_seeds == [10_000, 10_001, 10_002, 10_003, 10_004]
    assert args.output_csv.name == "lunar_lander_pymunk_ppo.csv"
    assert args.episode_output_csv.name == "lunar_lander_pymunk_ppo_episodes.csv"
    assert args.summary_output_csv.name == "lunar_lander_pymunk_ppo_summary.csv"
    assert args.manifest_json.name == "lunar_lander_pymunk_ppo_manifest.json"
    assert args.output_png.name == "lunar_lander_pymunk_ppo.png"


def test_parse_args_accepts_dqn():
    args = compare.parse_args(["--algorithm", "dqn"])

    assert args.algorithm == "dqn"
    assert args.output_csv.name == "lunar_lander_pymunk_dqn.csv"
    assert args.episode_output_csv.name == "lunar_lander_pymunk_dqn_episodes.csv"
    assert args.output_png.name == "lunar_lander_pymunk_dqn.png"
    assert args.dqn_learning_rate == 6.3e-4
    assert args.dqn_batch_size == 128
    assert args.dqn_buffer_size == 50_000
    assert args.dqn_learning_starts == 0
    assert args.dqn_gamma == 0.99
    assert args.dqn_target_update_interval == 250
    assert args.dqn_train_freq == 4
    assert args.dqn_gradient_steps == -1
    assert args.dqn_exploration_fraction == 0.12
    assert args.dqn_exploration_final_eps == 0.1


def test_parse_args_accepts_sac_defaults_and_entropy_values():
    args = compare.parse_args(["--algorithm", "sac"])

    assert args.algorithm == "sac"
    assert compare.ACTION_MODE_BY_ALGORITHM[args.algorithm] == "continuous"
    assert args.output_csv.name == "lunar_lander_pymunk_sac.csv"
    assert args.sac_learning_rate == 3e-4
    assert args.sac_buffer_size == 1_000_000
    assert args.sac_learning_starts == 100
    assert args.sac_batch_size == 256
    assert args.sac_tau == 0.005
    assert args.sac_gamma == 0.99
    assert args.sac_train_freq == 1
    assert args.sac_gradient_steps == 1
    assert args.sac_ent_coef == "auto"
    assert args.sac_target_update_interval == 1
    assert args.sac_target_entropy == "auto"

    numeric = compare.parse_args(
        ["--algorithm", "sac", "--sac-ent-coef", "0.2", "--sac-target-entropy", "-2"]
    )
    assert numeric.sac_ent_coef == 0.2
    assert numeric.sac_target_entropy == -2.0


def test_parse_args_rejects_invalid_entropy_value():
    with pytest.raises(SystemExit):
        compare.parse_args(["--sac-ent-coef", "adaptive"])


def test_episode_output_csv_is_derived_from_output_csv():
    args = compare.parse_args(["--output-csv", "results/comparison.csv"])

    assert args.episode_output_csv == Path("results/comparison_episodes.csv")


def test_explicit_episode_output_csv_is_preserved():
    args = compare.parse_args(["--episode-output-csv", "episodes/custom.csv"])

    assert args.episode_output_csv == Path("episodes/custom.csv")


def test_training_diagnostics_csv_path_is_optional_and_preserved():
    assert compare.parse_args([]).training_diagnostics_csv is None

    args = compare.parse_args(
        ["--training-diagnostics-csv", "diagnostics/ppo_updates.csv"]
    )

    assert args.training_diagnostics_csv == Path("diagnostics/ppo_updates.csv")


def test_parse_args_accepts_run_type_and_external_evaluation_seeds():
    args = compare.parse_args(
        [
            "--run-type",
            "pilot",
            "--eval-episodes",
            "99",
            "--evaluation-seeds",
            "31",
            "7",
            "19",
        ]
    )

    assert args.run_type == "pilot"
    assert args.evaluation_seeds == [31, 7, 19]
    assert args.eval_episodes == 3


def test_settle_diagnostic_is_disabled_by_default():
    args = compare.parse_args([])

    assert args.post_landing_settle_steps == 0
    assert args.post_landing_settle_timesteps == []
    assert args.settle_output_csv is None


def test_settle_output_csv_is_derived_when_diagnostic_is_requested():
    args = compare.parse_args(
        [
            "--output-csv",
            "results/comparison.csv",
            "--post-landing-settle-steps",
            "10",
            "--post-landing-settle-timesteps",
            "500",
            "1000",
        ]
    )

    assert args.settle_output_csv == Path("results/comparison_settle.csv")


def test_parse_args_accepts_diagnostic_options(tmp_path):
    args = compare.parse_args(
        [
            "--engines",
            "pymunk",
            "--checkpoint-freq",
            "100",
            "--checkpoint-dir",
            str(tmp_path),
            "--record-checkpoint-videos",
        ]
    )

    assert args.engines == ["pymunk"]
    assert args.checkpoint_freq == 100
    assert args.checkpoint_dir == tmp_path
    assert args.record_checkpoint_videos


@pytest.mark.parametrize(
    ("engine", "factory", "environment_id", "backend_module"),
    [
        (
            "box2d",
            compare.make_box2d_env,
            "LunarLander-v3",
            "gymnasium.envs.box2d.lunar_lander",
        ),
        (
            "pymunk",
            compare.make_pymunk_env,
            "LunarLander-v4",
            "gymnasium.envs.pymunk.lunar_lander",
        ),
    ],
)
def test_production_factories_use_registered_time_limit(
    engine, factory, environment_id, backend_module
):
    """Production benchmark factories use matching registered contracts."""
    if engine == "box2d":
        pytest.importorskip("Box2D")
    env = factory()
    try:
        description = compare.describe_environment(env)
        assert description["id"] == environment_id
        assert description["effective_max_episode_steps"] == 1_000
        assert [wrapper["class"] for wrapper in description["wrapper_chain"]] == [
            "TimeLimit",
            "OrderEnforcing",
        ]
        assert description["base_class"] == "LunarLander"
        assert description["base_module"] == backend_module
        assert description["internal_max_episode_steps"] is None
    finally:
        env.close()


@pytest.mark.parametrize(
    ("engine", "factory", "environment_id", "backend_module"),
    [
        (
            "box2d",
            compare.make_box2d_env,
            "LunarLanderContinuous-v3",
            "gymnasium.envs.box2d.lunar_lander",
        ),
        (
            "pymunk",
            compare.make_pymunk_env,
            "LunarLanderContinuous-v4",
            "gymnasium.envs.pymunk.lunar_lander",
        ),
    ],
)
def test_sac_factories_use_registered_continuous_time_limit(
    engine, factory, environment_id, backend_module
):
    if engine == "box2d":
        pytest.importorskip("Box2D")
    env = factory(algorithm="sac")
    try:
        description = compare.describe_environment(env)
        assert description["id"] == environment_id
        assert description["base_module"] == backend_module
        assert description["effective_max_episode_steps"] == 1_000
        assert description["constructor_arguments"]["continuous"] is True
        assert description["action_space"] == {
            "class": "Box",
            "module": "gymnasium.spaces.box",
            "shape": [2],
            "dtype": "float32",
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        }
    finally:
        env.close()


def test_algorithm_action_space_compatibility_is_validated_early():
    discrete = compare.make_pymunk_env()
    continuous = compare.make_pymunk_env(algorithm="sac")
    try:
        compare.validate_algorithm_action_space("ppo", discrete.action_space)
        compare.validate_algorithm_action_space("dqn", discrete.action_space)
        compare.validate_algorithm_action_space("sac", continuous.action_space)
        with pytest.raises(ValueError, match="continuous"):
            compare.validate_algorithm_action_space("sac", discrete.action_space)
        with pytest.raises(ValueError, match="discrete"):
            compare.validate_algorithm_action_space("dqn", continuous.action_space)
    finally:
        discrete.close()
        continuous.close()


def test_physics_diagnostic_factories_are_intentionally_unwrapped():
    """State-level diagnostics deliberately omit registration wrappers."""
    pytest.importorskip("Box2D")
    from scripts import analyze_lunar_lander_angular_dynamics as angular
    from scripts import compare_lunar_lander_fixed_actions as fixed
    from scripts import sweep_pymunk_lunar_lander_solver_iterations as sweep

    environments = [
        fixed.make_box2d_env(),
        fixed.make_pymunk_env(),
        angular.make_unwrapped_box2d_env(),
        angular.make_unwrapped_pymunk_env(),
    ]
    try:
        assert all(env.unwrapped is env for env in environments)
        assert all(not hasattr(env, "_max_episode_steps") for env in environments)
        assert sweep.make_unwrapped_box2d_env is angular.make_unwrapped_box2d_env
        assert sweep.make_unwrapped_pymunk_env is angular.make_unwrapped_pymunk_env
    finally:
        for env in environments:
            env.close()


@pytest.mark.parametrize(
    ("observation", "reward", "terminated", "truncated", "expected"),
    [
        (np.zeros(8), 100.0, True, False, "landing"),
        (np.zeros(8), -100.0, True, False, "crash"),
        (
            np.array([1.0, 0, 0, 0, 0, 0, 0, 0]),
            -100.0,
            True,
            False,
            "ambiguous_failure",
        ),
        (np.zeros(8), 0.0, False, True, "time_limit"),
    ],
)
def test_fixed_action_outcomes_use_engine_neutral_outputs(
    observation, reward, terminated, truncated, expected
):
    from scripts.compare_lunar_lander_fixed_actions import classify_outcome

    assert classify_outcome(observation, reward, terminated, truncated) == expected


def test_evaluate_policy_uses_engine_neutral_landing_classification():
    reset_seeds = []

    class Model:
        def predict(self, observation, deterministic):
            return 0, None

    class Env:
        action_space = None

        @property
        def unwrapped(self):
            return self

        def reset(self, seed):
            reset_seeds.append(seed)
            return np.zeros(8), {}

        def step(self, action):
            return np.zeros(8), 250.0, True, False, {}

        def close(self):
            pass

    result = compare.evaluate_policy(
        Model(), Env, seed=10_000, episodes=2, success_return_threshold=200
    )

    assert reset_seeds == [10_000, 10_001]
    assert result.return_threshold_success_rate == 1.0
    assert result.environment_success_rate == 1.0
    assert result.landing_rate == 1.0
    assert result.episodes[0]["return_threshold_success"] is True
    assert result.episodes[0]["environment_success"] is True


@pytest.mark.parametrize(
    ("observation", "reward", "terminated", "truncated", "expected"),
    [
        (np.zeros(8), 100.0, True, False, "landing"),
        (np.zeros(8), -100.0, True, False, "crash"),
        (
            np.array([1.0, 0, 0, 0, 0, 0, 0, 0]),
            -100.0,
            True,
            False,
            "ambiguous_failure",
        ),
        (np.zeros(8), 0.0, False, True, "time_limit"),
    ],
)
def test_classify_episode_is_engine_neutral(
    observation, reward, terminated, truncated, expected
):
    assert (
        compare.classify_episode(observation, reward, terminated, truncated) == expected
    )


def test_terminated_takes_precedence_over_truncated():
    assert compare.classify_episode(np.zeros(8), 100.0, True, True) == "landing"


def test_viewport_failure_is_engine_neutral_and_ambiguous():
    observation = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])

    assert (
        compare.classify_episode(observation, -100.0, True, False)
        == "ambiguous_failure"
    )


def test_explicit_evaluation_seeds_are_shared_verbatim():
    reset_seeds = []

    class Model:
        def predict(self, observation, deterministic):
            return 0, None

    class Env:
        prev_shaping = None

        @property
        def unwrapped(self):
            return self

        def reset(self, seed):
            reset_seeds.append(seed)
            return np.zeros(8), {}

        def step(self, action):
            return np.zeros(8), -100.0, True, False, {}

        def close(self):
            pass

    result = compare.evaluate_policy(
        Model(),
        Env,
        seed=None,
        episodes=None,
        success_return_threshold=200,
        evaluation_seeds=[91, 17, 42],
        bootstrap_samples=10,
    )

    assert reset_seeds == [91, 17, 42]
    assert [episode["evaluation_seed"] for episode in result.episodes] == [91, 17, 42]


@pytest.mark.parametrize(
    ("action", "main_penalty", "side_penalty"),
    [
        (np.array([0.0, 0.0], dtype=np.float32), 0.0, 0.0),
        (np.array([1.0, 0.0], dtype=np.float32), -0.3, 0.0),
        (np.array([0.0, -1.0], dtype=np.float32), 0.0, -0.03),
        (np.array([0.0, 1.0], dtype=np.float32), 0.0, -0.03),
    ],
)
def test_continuous_reward_decomposition_uses_public_engine_powers(
    action, main_penalty, side_penalty
):
    initial = np.zeros(8, dtype=np.float32)
    following = np.array([0.1, 0.2, 0.01, -0.02, 0.03, 0.0, 0.0, 0.0])
    decomposition = compare.RewardDecomposition(
        compare.RewardDecomposition.shaping(initial)
    )
    expected_shaping = float(
        np.sum(
            compare.RewardDecomposition.shaping(following)
            - compare.RewardDecomposition.shaping(initial)
        )
    )
    reward = expected_shaping + main_penalty + side_penalty

    decomposition.add_step(following, action, reward, terminated=False)

    components = decomposition.as_dict()
    assert components["main_engine_penalty"] == pytest.approx(main_penalty)
    assert components["side_engine_penalty"] == pytest.approx(side_penalty)
    assert sum(components.values()) == pytest.approx(reward)


def test_continuous_action_diagnostics_have_stable_schema_and_values():
    actions = [
        np.array([0.0, 0.0]),
        np.array([1.0, -0.75]),
        np.array([0.5, 1.0]),
        np.array([-1.0, 0.5]),
    ]

    diagnostics = compare.continuous_action_diagnostics(actions)

    assert tuple(diagnostics) == compare.CONTINUOUS_ACTION_DIAGNOSTIC_KEYS
    assert diagnostics["main_engine_activation_count"] == 2
    assert diagnostics["mean_active_main_power"] == pytest.approx(0.875)
    assert diagnostics["left_side_engine_activation_count"] == 1
    assert diagnostics["right_side_engine_activation_count"] == 1
    assert diagnostics["mean_active_side_power"] == pytest.approx(0.875)
    assert diagnostics["action_0_mean"] == pytest.approx(0.125)
    assert diagnostics["action_1_mean"] == pytest.approx(0.1875)
    assert diagnostics["boundary_saturation_frequency"] == pytest.approx(0.75)


def test_continuous_evaluation_and_settling_preserve_vector_actions():
    actions_seen = []

    class Model:
        def predict(self, observation, deterministic):
            assert deterministic is True
            return np.array([0.25, -0.75], dtype=np.float32), None

    class Env:
        prev_shaping = None

        @property
        def unwrapped(self):
            return self

        def reset(self, seed):
            return np.zeros(8, dtype=np.float32), {}

        def step(self, action):
            actions_seen.append(np.asarray(action).copy())
            return np.zeros(8, dtype=np.float32), 100.0, True, False, {}

        def close(self):
            pass

    result = compare.evaluate_policy(
        Model(),
        Env,
        seed=None,
        episodes=None,
        success_return_threshold=200,
        evaluation_seeds=[9],
        post_landing_settle_steps=2,
        action_mode="continuous",
    )

    np.testing.assert_array_equal(actions_seen[0], [0.25, -0.75])
    np.testing.assert_array_equal(actions_seen[1], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(actions_seen[2], np.zeros(2, dtype=np.float32))
    assert result.action_counts == "{}"
    diagnostics = json.loads(result.continuous_action_diagnostics)
    assert tuple(diagnostics) == tuple(
        sorted(compare.CONTINUOUS_ACTION_DIAGNOSTIC_KEYS)
    )


def test_continuous_video_uses_registered_factory_and_vector_action(
    monkeypatch, tmp_path
):
    actions = []
    created_ids = []

    class Model:
        def predict(self, observation, deterministic):
            return np.array([0.0, 0.0], dtype=np.float32), None

    def factory(render_mode=None):
        env = compare.make_pymunk_env(render_mode=render_mode, algorithm="sac")
        created_ids.append(env.spec.id)
        original_step = env.step

        def step(action):
            actions.append(np.asarray(action).copy())
            return original_step(action)

        env.step = step
        return env

    monkeypatch.setattr(compare, "write_video", lambda frames, path, fps: None)
    compare.record_policy_video(Model(), factory, tmp_path / "video.mp4", 7, 30, 1)

    assert created_ids == ["LunarLanderContinuous-v4"]
    assert len(actions) == 1
    assert actions[0].shape == (2,)


def test_post_landing_steps_do_not_change_evaluation_metrics(tmp_path):
    class Model:
        def predict(self, observation, deterministic):
            return 0, None

    class SuccessfulEnv:
        def __init__(self):
            self.steps = 0
            self.prev_shaping = None

        @property
        def unwrapped(self):
            return self

        def reset(self, seed):
            return np.zeros(8, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            observation = np.array(
                [
                    0.0,
                    0.073 if self.steps == 1 else 0.5,
                    0.01,
                    -0.01,
                    0.02,
                    0.03,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            if self.steps == 1:
                return observation, 100.0, True, False, {}
            return observation, 999.0, True, False, {}

        def close(self):
            pass

    result = compare.evaluate_policy(
        Model(),
        SuccessfulEnv,
        seed=10_000,
        episodes=1,
        success_return_threshold=200,
        post_landing_settle_steps=3,
    )

    assert result.episodes[0]["return"] == 100.0
    assert result.episodes[0]["length"] == 1
    assert result.episodes[0]["action_counts"] == "{0: 1}"
    assert json.loads(result.episodes[0]["final_observation"])[1] == pytest.approx(
        0.073
    )
    assert [row["settle_step"] for row in result.settle_steps] == [0, 1, 2, 3]

    output_csv = tmp_path / "settle.csv"
    settle_rows = [
        {
            "engine": "pymunk",
            "training_seed": 0,
            "training_timestep": 500,
            **row,
        }
        for row in result.settle_steps
    ]
    compare.append_settle_csv_rows(settle_rows, output_csv)
    with output_csv.open() as file:
        written_rows = list(csv.DictReader(file))

    assert len(written_rows) == 4
    assert float(written_rows[0]["y"]) == pytest.approx(0.073)
    assert written_rows[-1]["settle_step"] == "3"


def test_time_limit_episode_records_final_100_physics_steps():
    class Model:
        def predict(self, observation, deterministic):
            return 0, None

    class Physics:
        def __init__(self, env):
            self.env = env

        def physics_diagnostics(self, action):
            step = self.env.steps
            return {
                "action": action,
                "hull_linear_speed": step / 1000,
                "hull_angular_speed": 0.0,
                "left_leg_linear_speed": 0.0,
                "left_leg_angular_speed": 0.0,
                "right_leg_linear_speed": 0.0,
                "right_leg_angular_speed": 0.0,
                "hull_is_sleeping": False,
                "left_leg_is_sleeping": False,
                "right_leg_is_sleeping": False,
                "left_leg_contact": True,
                "right_leg_contact": True,
                "hull_kinetic_energy": 0.0,
                "total_kinetic_energy": 0.0,
                "idle_speed_threshold": 0.01,
                "sleep_time_threshold": 0.5,
                "left_rotary_limit_impulse": 0.1,
                "right_rotary_limit_impulse": 0.1,
                "left_motor_impulse": 0.8,
                "right_motor_impulse": 0.8,
            }

    class TimeLimitEnv:
        prev_shaping = None

        def __init__(self):
            self.steps = 0
            self.stable_landing_steps = 0
            self._physics = Physics(self)

        @property
        def unwrapped(self):
            return self

        def reset(self, seed):
            return np.zeros(8, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            truncated = self.steps == 1000
            return (
                np.zeros(8, dtype=np.float32),
                0.0,
                False,
                truncated,
                {},
            )

        def close(self):
            pass

    result = compare.evaluate_policy(
        Model(), TimeLimitEnv, seed=123, episodes=1, success_return_threshold=200
    )
    diagnostics = json.loads(result.episodes[0]["time_limit_final_100"])

    assert len(diagnostics) == 100
    assert diagnostics[0]["episode_step"] == 901
    assert diagnostics[-1]["episode_step"] == 1000
    assert diagnostics[-1]["action"] == 0
    assert diagnostics[-1]["stable_condition_counter"] == 0
    assert diagnostics[-1]["left_motor_impulse"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "make_env",
    [compare.make_box2d_env, compare.make_pymunk_env],
)
def test_episode_reward_decomposition_matches_return(make_env):
    pytest.importorskip("Box2D")

    class MainEnginePolicy:
        def predict(self, observation, deterministic):
            return 2, None

    result = compare.evaluate_policy(
        MainEnginePolicy(),
        make_env,
        seed=123,
        episodes=1,
        success_return_threshold=200,
    )
    episode = result.episodes[0]
    component_names = [
        "position_velocity_shaping",
        "angle_shaping",
        "leg_contact_shaping",
        "main_engine_penalty",
        "side_engine_penalty",
        "terminal_reward",
    ]

    assert sum(float(episode[name]) for name in component_names) == pytest.approx(
        episode["return"], abs=1e-3
    )
    assert episode["evaluation_seed"] == 123
    assert episode["length"] > 0
    assert episode["outcome"]
    assert episode["action_counts"]
    assert len(json.loads(episode["final_observation"])) == 8


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_make_algorithm_constructs_supported_algorithms(algorithm):
    pytest.importorskip("stable_baselines3")
    env = compare.make_pymunk_env()
    args = compare.parse_args(["--algorithm", algorithm, "--dqn-batch-size", "1"])

    model = compare.make_algorithm(args, env, seed=0)

    assert model.__class__.__name__.lower() == algorithm
    model.env.close()


def test_make_algorithm_constructs_sac_with_explicit_defaults():
    pytest.importorskip("stable_baselines3")
    env = compare.make_pymunk_env(algorithm="sac")
    args = _sac_args()

    model = compare.make_algorithm(args, env, seed=0)

    assert model.__class__.__name__ == "SAC"
    assert model.learning_rate == 3e-4
    assert model.buffer_size == 16
    assert model.learning_starts == 0
    assert model.batch_size == 2
    assert model.tau == 0.005
    assert model.gamma == 0.99
    assert model.gradient_steps == 1
    assert model.ent_coef == "auto"
    assert model.target_entropy == pytest.approx(-2.0)
    assert model.action_noise is None
    assert model.use_sde is False
    model.env.close()


def test_effective_sac_configuration_records_model_and_environment_contract():
    pytest.importorskip("stable_baselines3")
    env = compare.make_pymunk_env(algorithm="sac")
    args = _sac_args()
    model = compare.make_algorithm(args, env, seed=0)

    configuration = compare.effective_model_configuration(model, model.get_env(), args)

    assert configuration["model_class"] == "SAC"
    assert configuration["actor_class"] == "Actor"
    assert configuration["critic_class"] == "ContinuousCritic"
    assert configuration["actor_architecture"] == [256, 256]
    assert configuration["critic_architecture"] == [256, 256]
    assert configuration["critic_count"] == 2
    assert configuration["actor_optimizer_class"] == "Adam"
    assert configuration["critic_optimizer_class"] == "Adam"
    assert configuration["entropy_optimizer_class"] == "Adam"
    assert configuration["replay_buffer_class"] == "ReplayBuffer"
    assert configuration["replay_buffer_capacity"] == 16
    assert configuration["replay_buffer_handles_timeouts"] is True
    assert configuration["tau"] == 0.005
    assert configuration["requested_ent_coef"] == "auto"
    assert configuration["effective_ent_coef"] == pytest.approx(1.0)
    assert configuration["requested_target_entropy"] == "auto"
    assert configuration["effective_target_entropy"] == pytest.approx(-2.0)
    assert configuration["environment"]["id"] == "LunarLanderContinuous-v4"
    model.env.close()


def test_training_uses_one_uninterrupted_learn_call(monkeypatch):
    pytest.importorskip("stable_baselines3")
    args = _sac_args("--engines", "pymunk")
    calls = []

    class Model:
        def learn(self, total_timesteps, callback):
            calls.append((total_timesteps, callback.__class__.__name__))
            return self

    model = Model()
    monkeypatch.setattr(compare, "make_algorithm", lambda args, env, seed: model)

    rows, returned = compare.train_and_evaluate(
        "pymunk",
        lambda: compare.make_pymunk_env(algorithm="sac"),
        0,
        args,
    )

    assert calls == [(2, "PeriodicEvaluationCallback")]
    assert rows == []
    assert returned is model


@pytest.mark.parametrize(
    "make_env",
    [
        compare.make_box2d_env,
        compare.make_pymunk_env,
    ],
)
def test_dqn_predicts_actions_for_discrete_lunar_lander_envs(make_env):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("Box2D")
    env = make_env()
    args = _dqn_args()
    model = compare.make_algorithm(args, env, seed=0)

    observation, _ = env.reset(seed=0)
    action, _ = model.predict(observation, deterministic=True)

    assert env.action_space.contains(int(action))
    model.env.close()


def test_write_csv_includes_algorithm_column(tmp_path):
    output_csv = tmp_path / "results.csv"
    compare.write_csv(
        [
            {
                "algorithm": "dqn",
                "engine": "pymunk",
                "seed": 0,
                "timestep": 4,
                "mean_return": 0.0,
                "return_std": 0.0,
                "success_rate": 0.0,
                "mean_episode_length": 1.0,
                "termination_counts": "{}",
                "action_counts": "{}",
                "early_action_counts": "{}",
                "middle_action_counts": "{}",
                "late_action_counts": "{}",
            }
        ],
        output_csv,
    )

    with output_csv.open() as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["algorithm"] == "dqn"


def test_append_csv_row_preserves_previous_evaluations(tmp_path):
    output_csv = tmp_path / "results.csv"
    row = {
        "algorithm": "dqn",
        "engine": "pymunk",
        "seed": 0,
        "timestep": 4,
    }

    compare.append_csv_row(row, output_csv)
    compare.append_csv_row({**row, "timestep": 8}, output_csv)

    with output_csv.open() as file:
        rows = list(csv.DictReader(file))

    assert [int(result["timestep"]) for result in rows] == [4, 8]


def test_manifest_contains_reproducibility_fields():
    args = compare.parse_args(
        [
            "--algorithm",
            "dqn",
            "--run-type",
            "pilot",
            "--seeds",
            "3",
            "5",
            "--evaluation-seeds",
            "101",
            "102",
        ]
    )
    timestamp = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)

    manifest = compare.create_run_manifest(
        args,
        command=["python", "benchmark.py", "--algorithm", "dqn"],
        timestamp=timestamp,
    )

    assert manifest["schema_version"] == 5
    assert manifest["git_sha"]
    assert isinstance(manifest["dirty_worktree"], bool)
    assert manifest["status"] == "running"
    assert manifest["completed_pairs"] == []
    assert manifest["command"] == "python benchmark.py --algorithm dqn"
    assert manifest["python"]
    assert manifest["algorithm"] == "dqn"
    assert manifest["action_mode"] == "discrete"
    assert manifest["run_type"] == "pilot"
    assert manifest["training_seeds"] == [3, 5]
    assert manifest["evaluation_seeds"] == [101, 102]
    assert manifest["hyperparameters"]["buffer_size"] == 50_000
    assert manifest["environment_configuration"]["engines"] == [
        "box2d",
        "pymunk",
    ]
    environments = manifest["environment_configuration"]["environments"]
    assert environments["box2d"]["id"] == "LunarLander-v3"
    assert environments["pymunk"]["id"] == "LunarLander-v4"
    for environment in environments.values():
        assert environment["effective_max_episode_steps"] == 1_000
        assert [item["class"] for item in environment["wrapper_chain"]] == [
            "TimeLimit",
            "OrderEnforcing",
        ]
        assert environment["spec"]["max_episode_steps"] == 1_000
    assert {"gymnasium", "numpy", "pymunk", "stable-baselines3", "torch"} <= set(
        manifest["dependencies"]
    )


def test_sac_manifest_fingerprints_continuous_contract_and_configuration():
    args = _sac_args("--engines", "pymunk", "--evaluation-seeds", "17")

    manifest = compare.create_run_manifest(args)

    assert manifest["schema_version"] == 5
    assert manifest["algorithm"] == "sac"
    assert manifest["action_mode"] == "continuous"
    assert manifest["resolved_constructor"]["ent_coef"] == "auto"
    assert manifest["resolved_constructor"]["target_entropy"] == "auto"
    environment = manifest["environment_configuration"]["environments"]["pymunk"]
    assert environment["id"] == "LunarLanderContinuous-v4"
    assert environment["effective_max_episode_steps"] == 1_000
    assert environment["action_space"]["shape"] == [2]
    assert environment["action_space"]["low"] == [-1.0, -1.0]

    changed = _sac_args(
        "--engines",
        "pymunk",
        "--evaluation-seeds",
        "17",
        "--sac-tau",
        "0.01",
    )
    assert compare.run_configuration_fingerprint(
        args
    ) != compare.run_configuration_fingerprint(changed)


def test_summary_reports_windows_auc_per_seed_and_schema(tmp_path):
    args = compare.parse_args(
        [
            "--run-type",
            "pilot",
            "--seeds",
            "1",
            "2",
            "--evaluation-seeds",
            "101",
            "102",
            "--final-window-checkpoints",
            "2",
            "--bootstrap-samples",
            "20",
        ]
    )
    rows = []
    for seed, returns in ((1, [0.0, 10.0, 5.0]), (2, [2.0, 12.0, 7.0])):
        for timestep, mean_return in zip((0, 100, 200), returns, strict=True):
            rows.append(
                {
                    "algorithm": "ppo",
                    "run_type": "pilot",
                    "engine": "pymunk",
                    "seed": seed,
                    "timestep": timestep,
                    "mean_return": mean_return,
                    "landing_rate": 0.5,
                    "failure_rate": 0.25,
                    "crash_rate": 0.25,
                    "viewport_exit_rate": 0.0,
                    "time_limit_rate": 0.25,
                    "mean_episode_length": 100.0,
                }
            )

    summaries = compare.summarize_results(rows, args)
    output_csv = tmp_path / "summary.csv"
    compare.write_summary_csv(summaries, output_csv)

    assert summaries[0]["final_window_start"] == 100
    assert summaries[0]["best_window_start"] == 100
    assert json.loads(summaries[0]["final_per_seed_mean_return"]) == {
        "1": 7.5,
        "2": 9.5,
    }
    assert summaries[0]["normalized_auc_mean"] == pytest.approx(7.25)
    with output_csv.open() as file:
        written = list(csv.DictReader(file))
    assert list(written[0]) == compare.SUMMARY_CSV_FIELDNAMES
    assert written[0]["run_type"] == "pilot"
    assert written[0]["best_window_interpretation"] == "descriptive; selection-biased"


def test_summary_bootstraps_training_seeds_and_reports_paired_differences(monkeypatch):
    args = compare.parse_args(
        [
            "--seeds",
            "1",
            "2",
            "--evaluation-seeds",
            "10",
            "11",
            "--final-window-checkpoints",
            "1",
        ]
    )
    captured_lengths = []
    original = compare.bootstrap_confidence_interval

    def capture(values, samples, seed=0):
        captured_lengths.append(len(values))
        return original(values, 20, seed)

    monkeypatch.setattr(compare, "bootstrap_confidence_interval", capture)
    rows = []
    for engine, offset in (("box2d", 2.0), ("pymunk", 0.0)):
        for seed in args.seeds:
            for timestep in (0, 100):
                rows.append(
                    {
                        "engine": engine,
                        "seed": seed,
                        "timestep": timestep,
                        "mean_return": seed + offset + timestep / 100,
                        "landing_rate": 0.5 + offset / 10,
                        "failure_rate": 0.5 - offset / 10,
                        "crash_rate": 0.25,
                        "viewport_exit_rate": 0.0,
                        "time_limit_rate": 0.25,
                        "mean_episode_length": 100 - offset,
                    }
                )

    summaries = compare.summarize_results(rows, args)
    paired = next(
        row for row in summaries if row["comparison_type"].startswith("paired")
    )

    assert set(captured_lengths) == {2}
    assert json.loads(paired["final_per_seed_mean_return"]) == {"1": 2.0, "2": 2.0}
    assert paired["final_mean_failure_rate"] == pytest.approx(-0.2)
    assert paired["normalized_auc_mean"] == pytest.approx(2.0)


def test_normalized_auc_is_invariant_to_evaluation_frequency():
    sparse = compare.normalized_learning_curve_area([0, 100], [0, 100])
    dense = compare.normalized_learning_curve_area(
        [0, 25, 50, 75, 100], [0, 25, 50, 75, 100]
    )

    assert sparse == pytest.approx(50.0)
    assert dense == pytest.approx(sparse)


@pytest.mark.parametrize(
    ("timesteps", "values"),
    [
        ([0], [1]),
        ([0, 0], [1, 2]),
        ([0, 10], [1]),
        ([0, 10, 5], [1, 2, 3]),
        ([1, 10], [1, 2]),
    ],
)
def test_normalized_auc_rejects_incomplete_or_zero_width_curves(timesteps, values):
    with pytest.raises(ValueError, match="strictly ordered from step zero"):
        compare.normalized_learning_curve_area(timesteps, values)


@pytest.mark.parametrize(
    ("timesteps", "values", "expected"),
    [
        ([0, 10], [3, 3], 3.0),
        ([0, 5, 10], [0, 10, 0], 5.0),
        ([0.0, 0.25, 1.0], [-2.0, 2.0, 2.0], 1.5),
    ],
)
def test_normalized_auc_known_curves(timesteps, values, expected):
    assert compare.normalized_learning_curve_area(timesteps, values) == pytest.approx(
        expected
    )


def test_normalized_auc_treats_integer_and_float_timesteps_equally():
    values = [1.5, -2.0, 4.0]
    assert compare.normalized_learning_curve_area(
        [0, 3, 10], values
    ) == compare.normalized_learning_curve_area([0.0, 3.0, 10.0], values)


def test_normalized_auc_matches_numpy_2_trapezoid():
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        pytest.skip("NumPy 1.x does not provide np.trapezoid")
    timesteps = np.array([0, 3, 10, 25], dtype=np.float64)
    values = np.array([-4.5, 2.0, 7.25, -1.0], dtype=np.float64)

    assert compare.normalized_learning_curve_area(timesteps, values) == pytest.approx(
        float(trapezoid(values, timesteps) / timesteps[-1])
    )


def test_duplicate_result_keys_are_rejected():
    rows = [
        {"engine": "box2d", "seed": 1, "timestep": 10},
        {"engine": "box2d", "seed": 1, "timestep": 10},
    ]

    with pytest.raises(RuntimeError, match="Duplicate result key"):
        compare.deduplicate_rows(rows, ("engine", "seed", "timestep"))


def test_acceptance_rejects_dirty_tracked_source(monkeypatch):
    args = compare.parse_args(["--run-type", "acceptance"])

    def git_bytes(*arguments):
        if arguments == ("rev-parse", "HEAD"):
            return b"abc123\n"
        if arguments == ("ls-files", "--others", "--exclude-standard"):
            return b""
        return b"dirty diff"

    monkeypatch.setattr(compare, "_git_bytes", git_bytes)

    with pytest.raises(RuntimeError, match="clean tracked source"):
        compare.source_identity(args)


def test_checkpoint_and_video_paths_are_protected(tmp_path):
    args = compare.parse_args(
        [
            "--checkpoint-freq",
            "10",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--record-checkpoint-videos",
            "--record-videos",
            "--video-dir",
            str(tmp_path / "videos"),
            "--train-steps",
            "10",
            "--engines",
            "box2d",
        ]
    )
    artifacts = [path for path in compare.output_artifact_paths(args) if path]
    checkpoint = tmp_path / "checkpoints" / "ppo_box2d_seed_0_steps_10.zip"
    video = tmp_path / "videos" / "ppo_box2d_seed_0_steps_10_eval_seed_10000.mp4"

    assert checkpoint in artifacts
    assert video in artifacts
    checkpoint.parent.mkdir()
    checkpoint.touch()
    with pytest.raises(FileExistsError):
        compare.ensure_outputs_available(artifacts, overwrite=False)


def test_interrupted_run_resumes_only_incomplete_pair(monkeypatch, tmp_path):
    output = tmp_path / "results.csv"
    arguments = [
        "benchmark.py",
        "--algorithm",
        "dqn",
        "--engines",
        "box2d",
        "pymunk",
        "--train-steps",
        "10",
        "--eval-freq",
        "10",
        "--seeds",
        "3",
        "--evaluation-seeds",
        "101",
        "--output-csv",
        str(output),
        "--episode-output-csv",
        str(tmp_path / "episodes.csv"),
        "--summary-output-csv",
        str(tmp_path / "summary.csv"),
        "--manifest-json",
        str(tmp_path / "manifest.json"),
        "--output-png",
        str(tmp_path / "plot.png"),
    ]
    calls = []
    fail_pymunk = True

    class Model:
        _benchmark_settle_rows = []
        _benchmark_training_rows = []

        def get_env(self):
            return object()

    def fake_train(engine, make_env, seed, args, **kwargs):
        nonlocal fail_pymunk
        calls.append((engine, seed))
        if engine == "pymunk" and fail_pymunk:
            fail_pymunk = False
            raise RuntimeError("interrupted")
        model = Model()
        model._benchmark_episode_rows = []
        rows = []
        for timestep in (0, 10):
            rows.append(
                {
                    "algorithm": "dqn",
                    "run_type": "smoke",
                    "engine": engine,
                    "seed": seed,
                    "timestep": timestep,
                    "mean_return": 1.0,
                    "landing_rate": 0.5,
                    "failure_rate": 0.5,
                    "crash_rate": 0.5,
                    "viewport_exit_rate": 0.0,
                    "time_limit_rate": 0.0,
                    "mean_episode_length": 10.0,
                }
            )
        return rows, model

    monkeypatch.setattr(compare, "train_and_evaluate", fake_train)
    monkeypatch.setattr(compare, "effective_model_configuration", lambda *_: {})
    monkeypatch.setattr(
        compare, "write_plot", lambda rows, path: path.write_bytes(b"png")
    )
    original_parse_args = compare.parse_args
    monkeypatch.setattr(
        compare, "parse_args", lambda: original_parse_args(arguments[1:])
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        compare.main()
    failed = json.loads((tmp_path / "manifest.json").read_text())
    assert failed["status"] == "failed"
    assert failed["completed_pairs"] == [{"engine": "box2d", "training_seed": 3}]
    assert failed["current_pair"] == {"engine": "pymunk", "training_seed": 3}
    assert not (tmp_path / "summary.csv").exists()

    arguments.append("--resume")
    compare.main()
    completed = json.loads((tmp_path / "manifest.json").read_text())
    assert calls == [("box2d", 3), ("pymunk", 3), ("pymunk", 3)]
    assert completed["status"] == "completed"
    assert completed["current_pair"] is None
    assert completed["completed_at"]
    with output.open() as file:
        assert len(list(csv.DictReader(file))) == 4


def test_resume_rejects_incompatible_configuration(monkeypatch, tmp_path):
    args = compare.parse_args(
        [
            "--output-csv",
            str(tmp_path / "results.csv"),
            "--manifest-json",
            str(tmp_path / "manifest.json"),
        ]
    )
    manifest = compare.create_run_manifest(args)
    compare.write_manifest(manifest, args.manifest_json)
    argv = [
        "benchmark.py",
        "--resume",
        "--train-steps",
        "999",
        "--output-csv",
        str(args.output_csv),
        "--manifest-json",
        str(args.manifest_json),
    ]
    original_parse_args = compare.parse_args
    monkeypatch.setattr(compare, "parse_args", lambda: original_parse_args(argv[1:]))

    with pytest.raises(RuntimeError, match="different run configuration"):
        compare.main()


def test_resume_rejects_schema_four_with_clear_message(monkeypatch, tmp_path):
    output = tmp_path / "results.csv"
    manifest_path = tmp_path / "manifest.json"
    args = compare.parse_args(
        ["--output-csv", str(output), "--manifest-json", str(manifest_path)]
    )
    manifest = compare.create_run_manifest(args)
    manifest["schema_version"] = 4
    compare.write_manifest(manifest, manifest_path)
    original_parse_args = compare.parse_args
    monkeypatch.setattr(
        compare,
        "parse_args",
        lambda: original_parse_args(
            [
                "--resume",
                "--output-csv",
                str(output),
                "--manifest-json",
                str(manifest_path),
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="schema 4 runs cannot be migrated"):
        compare.main()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("id", "DifferentLunarLander-v99"),
        ("wrapper_chain", [{"class": "DifferentWrapper"}]),
        ("effective_max_episode_steps", 999),
    ],
)
def test_resume_rejects_changed_environment_contract(
    monkeypatch, tmp_path, field, replacement
):
    args = compare.parse_args(
        [
            "--engines",
            "pymunk",
            "--output-csv",
            str(tmp_path / "results.csv"),
            "--manifest-json",
            str(tmp_path / "manifest.json"),
        ]
    )
    manifest = compare.create_run_manifest(args)
    compare.write_manifest(manifest, args.manifest_json)
    original_configuration = compare.benchmark_environment_configuration

    def changed_configuration(parsed_args):
        configuration = original_configuration(parsed_args)
        configuration["environments"]["pymunk"][field] = replacement
        return configuration

    monkeypatch.setattr(
        compare, "benchmark_environment_configuration", changed_configuration
    )
    argv = [
        "benchmark.py",
        "--resume",
        "--engines",
        "pymunk",
        "--output-csv",
        str(args.output_csv),
        "--manifest-json",
        str(args.manifest_json),
    ]
    original_parse_args = compare.parse_args
    monkeypatch.setattr(compare, "parse_args", lambda: original_parse_args(argv[1:]))

    with pytest.raises(RuntimeError, match="different run configuration"):
        compare.main()


def test_existing_outputs_require_explicit_overwrite(tmp_path):
    output = tmp_path / "existing.csv"
    output.write_text("existing\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        compare.ensure_outputs_available([output], overwrite=False)

    compare.ensure_outputs_available([output], overwrite=True)


def test_append_training_diagnostics_row_writes_incrementally(tmp_path):
    output_csv = tmp_path / "nested" / "ppo_updates.csv"
    row = {
        "engine": "pymunk",
        "training_seed": 7,
        "timestep": 256,
        "n_updates": 4,
        "approx_kl": 0.01,
        "rollout_episode_return": -100.0,
        "rollout_episode_length": 200.0,
    }

    compare.append_training_diagnostics_row(row, output_csv)
    with output_csv.open() as file:
        first_rows = list(csv.DictReader(file))
    assert [int(result["timestep"]) for result in first_rows] == [256]

    compare.append_training_diagnostics_row({**row, "timestep": 512}, output_csv)
    with output_csv.open() as file:
        rows = list(csv.DictReader(file))

    assert [int(result["timestep"]) for result in rows] == [256, 512]
    assert rows[0]["engine"] == "pymunk"
    assert float(rows[0]["approx_kl"]) == pytest.approx(0.01)


def test_ppo_training_diagnostics_records_every_policy_update(tmp_path):
    pytest.importorskip("stable_baselines3")
    diagnostics_csv = tmp_path / "ppo_updates.csv"
    args = compare.parse_args(
        [
            "--train-steps",
            "8",
            "--eval-freq",
            "8",
            "--eval-episodes",
            "1",
            "--n-steps",
            "4",
            "--batch-size",
            "4",
            "--n-epochs",
            "1",
            "--training-diagnostics-csv",
            str(diagnostics_csv),
        ]
    )

    _, model = compare.train_and_evaluate(
        "pymunk",
        compare.make_pymunk_env,
        seed=0,
        args=args,
        training_diagnostics_csv=diagnostics_csv,
    )

    with diagnostics_csv.open() as file:
        rows = list(csv.DictReader(file))
    assert [int(row["timestep"]) for row in rows] == [4, 8]
    assert all(row["approx_kl"] for row in rows)
    assert all(row["learning_rate"] for row in rows)
    model.env.close()


def test_small_dqn_smoke_run_on_pymunk(tmp_path):
    pytest.importorskip("stable_baselines3")
    args = _dqn_args()
    episode_output_csv = tmp_path / "episodes.csv"

    rows, model = compare.train_and_evaluate(
        "pymunk",
        compare.make_pymunk_env,
        seed=0,
        args=args,
        episode_output_csv=episode_output_csv,
    )

    assert len(rows) == 2
    assert rows[0]["algorithm"] == "dqn"
    assert rows[0]["engine"] == "pymunk"
    assert [row["timestep"] for row in rows] == [0, 4]
    assert episode_output_csv.exists()
    with episode_output_csv.open() as file:
        episode_rows = list(csv.DictReader(file))
    assert len(episode_rows) == 2
    reward_components = [
        "position_velocity_shaping",
        "angle_shaping",
        "leg_contact_shaping",
        "main_engine_penalty",
        "side_engine_penalty",
        "terminal_reward_override",
    ]
    assert sum(float(episode_rows[0][name]) for name in reward_components) == (
        pytest.approx(float(episode_rows[0]["episode_return"]), abs=1e-3)
    )
    model.env.close()
