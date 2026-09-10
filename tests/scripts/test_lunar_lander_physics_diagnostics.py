"""Tests for LunarLander research diagnostics kept outside environment tests."""

import json

import numpy as np
import pytest

pytest.importorskip("pymunk")
pytest.importorskip("Box2D")

from scripts.analyze_lunar_lander_angular_dynamics import initial_rows  # noqa: E402
from scripts.pymunk_lunar_lander_terrain import (  # noqa: E402
    DiagnosticPymunkLunarLanderDemo,
)
from scripts.pymunk_lunar_lander_terrain import (  # noqa: E402
    physics_diagnostics as script_physics_diagnostics,
)
from scripts.sweep_pymunk_lunar_lander_solver_iterations import (  # noqa: E402
    aggregate_scores,
    sweep,
)
from tests.envs.pymunk_lunar_lander_test_helpers import (  # noqa: E402
    DiagnosticPymunkLunarLanderDemo as _TestDiagnosticDemo,
)
from tests.envs.pymunk_lunar_lander_test_helpers import (  # noqa: E402
    physics_diagnostics as _test_physics_diagnostics,
)


def test_selected_solver_improves_trajectory_error_over_30_iterations():
    """Keep the calibrated solver ranking covered as research tooling."""
    scores = aggregate_scores(sweep([30, 180], range(100, 102), steps=100))

    assert scores[180] < scores[30]


def test_matched_seed_initial_state_distribution_moments_match_box2d():
    """Compare reset distribution moments over matched engine seeds."""
    rows = initial_rows(1000)
    box_rows = [row for row in rows if row["engine"] == "box2d"]
    pymunk_rows = [row for row in rows if row["engine"] == "pymunk"]

    box_vx = np.array([json.loads(row["hull_velocity"])[0] for row in box_rows])
    pymunk_vx = np.array([json.loads(row["hull_velocity"])[0] for row in pymunk_rows])
    box_omega = np.array([row["hull_angular_velocity"] for row in box_rows])
    pymunk_omega = np.array([row["hull_angular_velocity"] for row in pymunk_rows])
    box_vy = np.array([json.loads(row["hull_velocity"])[1] for row in box_rows])
    pymunk_vy = np.array([json.loads(row["hull_velocity"])[1] for row in pymunk_rows])
    box_angle = np.array([row["hull_angle"] for row in box_rows])
    pymunk_angle = np.array([row["hull_angle"] for row in pymunk_rows])

    assert np.std(pymunk_vx) == pytest.approx(np.std(box_vx), rel=0.02)
    assert np.std(pymunk_vy) == pytest.approx(np.std(box_vy), rel=0.02)
    assert np.std(pymunk_angle) == pytest.approx(np.std(box_angle), rel=0.25)
    assert np.std(pymunk_omega) == pytest.approx(np.std(box_omega), rel=0.25)
    assert np.mean(pymunk_vx) == pytest.approx(np.mean(box_vx), abs=0.01)
    assert np.mean(pymunk_vy) == pytest.approx(np.mean(box_vy), abs=0.1)
    assert pymunk_rows[0]["total_mass"] == pytest.approx(
        box_rows[0]["total_mass"], rel=1e-6
    )
    assert pymunk_rows[0]["articulated_inertia"] == pytest.approx(
        box_rows[0]["articulated_inertia"], rel=0.01
    )


def test_test_local_telemetry_matches_script_measurements():
    """Ensure test-local telemetry retains the established measurements."""
    script_demo = DiagnosticPymunkLunarLanderDemo(seed=123)
    test_demo = _TestDiagnosticDemo(seed=123)
    for action in [0, 2, 1, 3, 0]:
        script_state = script_demo.step(action)
        test_state = test_demo.step(action)
        assert np.array_equal(script_state.as_array(), test_state.as_array())
        assert script_physics_diagnostics(
            script_demo, action
        ) == _test_physics_diagnostics(test_demo, action)
        assert script_demo.last_engine_diagnostics == test_demo.last_engine_diagnostics
