import pytest

from gymnasium.envs.toy_text.taxi import TaxiEnv


@pytest.mark.parametrize("is_rainy", [False, True])
def test_taxi_action_mask(is_rainy):
    env = TaxiEnv(is_rainy=is_rainy)

    for state in env.P:
        mask = env.action_mask(state)
        for action, possible in enumerate(mask):
            transitions = env.P[state][action]
            if possible:
                _, next_state, _, _ = transitions[0]
                assert state != next_state
            else:
                for _, next_state, _, _ in transitions:
                    assert state == next_state


def test_taxi_encode_decode_roundtrip():
    env = TaxiEnv()

    for state in env.P:
        assert env.encode(*env.decode(state)) == state


def test_taxi_state_counts():
    """Total, initial, and reachable state counts derived from `P`.

    Matches the docstring counts: 500 total, 300 initial, 404 reachable
    (300 initial + 100 in-taxi + 4 post-dropoff terminals).
    """
    env = TaxiEnv()
    assert len(env.P) == 500

    initial_states = {s for s in env.P if env.initial_state_distrib[s] > 0}
    assert len(initial_states) == 300

    reachable = set(initial_states)
    frontier = set(initial_states)
    while frontier:
        next_frontier = set()
        for state in frontier:
            for transitions in env.P[state].values():
                for _prob, next_state, _reward, done in transitions:
                    if next_state not in reachable:
                        reachable.add(next_state)
                        if not done:
                            next_frontier.add(next_state)
        frontier = next_frontier
    assert len(reachable) == 404


@pytest.mark.parametrize("is_rainy", [False, True])
def test_taxi_disallowed_transitions(is_rainy):
    """No transition in P crosses an interior wall, with or without `is_rainy`."""
    disallowed_transitions = {
        ((0, 1), (0, 3)),
        ((0, 3), (0, 1)),
        ((1, 0), (1, 2)),
        ((1, 2), (1, 0)),
        ((3, 1), (3, 3)),
        ((3, 3), (3, 1)),
        ((3, 3), (3, 5)),
        ((3, 5), (3, 3)),
        ((4, 1), (4, 3)),
        ((4, 3), (4, 1)),
        ((4, 3), (4, 5)),
        ((4, 5), (4, 3)),
    }
    env = TaxiEnv(is_rainy=is_rainy)

    for state, state_dict in env.P.items():
        start_row, start_col, _, _ = env.decode(state)
        for transitions in state_dict.values():
            for _, new_state, _, _ in transitions:
                end_row, end_col, _, _ = env.decode(new_state)
                assert (
                    (start_row, start_col),
                    (end_row, end_col),
                ) not in disallowed_transitions


@pytest.mark.parametrize(
    "kwargs,movement_probs",
    [
        pytest.param({}, [1.0], id="dry"),
        pytest.param({"is_rainy": True}, [0.8, 0.1, 0.1], id="rainy"),
        pytest.param(
            {"is_rainy": True, "rainy_probability": 0.5},
            [0.5, 0.25, 0.25],
            id="rainy-p=0.5",
        ),
        pytest.param(
            {"is_rainy": True, "rainy_probability": 1.0},
            [1.0, 0.0, 0.0],
            id="rainy-p=1.0",
        ),
    ],
)
def test_transition_probabilities(kwargs, movement_probs):
    """Every (state, action) entry in P has the expected probability tuple.

    Movement actions (0-3) carry `movement_probs` (ordering: ahead, left, right).
    Pickup/dropoff (4, 5) always carry [1.0]. `fickle_passenger` does not touch
    P — its cases here just confirm the tunable rainy behaviour is unaffected.
    """
    env = TaxiEnv(**kwargs)
    for state_dict in env.P.values():
        for action, transitions in state_dict.items():
            probs = [t[0] for t in transitions]
            expected = movement_probs if action <= 3 else [1.0]
            assert probs == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs,valid_probs",
    [
        pytest.param({}, [1.0], id="dry"),
        pytest.param({"is_rainy": True}, [0.8, 0.1], id="rainy"),
    ],
)
def test_step_info_prob(kwargs, valid_probs):
    """`info["prob"]` returned from step() matches one of the P-branch probabilities."""
    env = TaxiEnv(**kwargs)
    env.reset()
    _, _, _, _, info = env.step(0)
    assert any(info["prob"] == pytest.approx(p) for p in valid_probs)


def _lateral_positions(env, row, col, action, pass_idx=0, dest_idx=1):
    """Return the destination (row, col) for each of the 3 rainy transitions.

    Returns a tuple (ahead, left, right) matching the order in env.P.
    """
    state = env.encode(row, col, pass_idx, dest_idx)
    transitions = env.P[state][action]
    assert len(transitions) == 3, (
        f"Expected 3 rainy transitions, got {len(transitions)}"
    )
    return tuple(
        tuple(env.decode(new_state))[:2]  # (new_row, new_col)
        for _prob, new_state, _reward, _done in transitions
    )


@pytest.fixture
def rainy_env():
    return TaxiEnv(is_rainy=True)


@pytest.mark.parametrize(
    "row,col,action,ahead,left,right",
    [
        # Directional convention: left/right relative to heading, from (1, 3).
        #   south → left=east,  right=west
        #   north → left=west,  right=east
        #   east  → left=north, right=south
        #   west  → left=south, right=north
        pytest.param(1, 3, 0, (2, 3), (1, 4), (1, 2), id="south-lateral"),
        pytest.param(1, 3, 1, (0, 3), (1, 2), (1, 4), id="north-lateral"),
        pytest.param(1, 3, 2, (1, 4), (0, 3), (2, 3), id="east-lateral"),
        pytest.param(1, 3, 3, (1, 2), (2, 3), (0, 3), id="west-lateral"),
        # wall variant: N/S lateral must not be gated by the E/W wall check.
        # At (1, 4) going west, the south lateral to (2, 4) is open even though the
        # outer grid wall sits at desc[*, 10]; at (2, 1) going east, the north
        # lateral to (1, 1) is open even though desc[2, 4] = '|' (east wall).
        pytest.param(
            1, 4, 3, (1, 3), (2, 4), (0, 4), id="west-south-lateral-at-right-edge"
        ),
        pytest.param(
            2, 1, 2, (2, 2), (1, 1), (3, 1), id="east-north-lateral-past-east-wall"
        ),
        # when the primary move is blocked (wall or boundary), no drift.
        pytest.param(4, 3, 0, (4, 3), (4, 3), (4, 3), id="south-boundary-blocked"),
        pytest.param(0, 3, 1, (0, 3), (0, 3), (0, 3), id="north-boundary-blocked"),
        pytest.param(0, 1, 2, (0, 1), (0, 1), (0, 1), id="east-wall-blocked"),
        pytest.param(0, 2, 3, (0, 2), (0, 2), (0, 2), id="west-wall-blocked"),
        # Lateral that would pass through an interior wall stays put.
        # At (0, 2) going south, the west lateral to (0, 1) is blocked by
        # desc[1, 4] = '|' while the primary and east lateral remain open.
        pytest.param(
            0, 2, 0, (1, 2), (0, 3), (0, 2), id="south-west-lateral-wall-blocked"
        ),
    ],
)
def test_rainy_transition_targets(rainy_env, row, col, action, ahead, left, right):
    """Validate rainy (ahead, left, right) target cells across directional, wall, and boundary cases.

    Covers left/right convention and N/S laterals not gated by E/W wall
    checks, no lateral drift when the primary move is blocked, and
    the lateral-into-interior-wall case.
    """
    got_ahead, got_left, got_right = _lateral_positions(rainy_env, row, col, action)
    assert got_ahead == ahead
    assert got_left == left
    assert got_right == right


@pytest.mark.parametrize(
    "fickle_probability,should_change",
    [
        pytest.param(1.0, True, id="p=1.0-always-changes"),
        pytest.param(0.0, False, id="p=0.0-never-changes"),
    ],
)
def test_fickle_probability_extremes(fickle_probability, should_change):
    """Fickle at the extremes: p=1.0 always changes destination, p=0.0 never does."""
    env = TaxiEnv(fickle_passenger=True, fickle_probability=fickle_probability)
    env.reset(seed=0)
    _, _, pass_idx, orig_dest_idx = env.decode(env.s)
    env.s = env.encode(*env.locs[pass_idx], pass_idx, orig_dest_idx)
    env.step(4)  # pickup at the passenger's source
    # Movement direction that's guaranteed not to be a no-op from any source loc.
    action = 0 if env.locs[pass_idx][0] == 0 else 1
    state, *_ = env.step(action)
    _, _, _, new_dest_idx = env.decode(state)
    assert (new_dest_idx != orig_dest_idx) == should_change
