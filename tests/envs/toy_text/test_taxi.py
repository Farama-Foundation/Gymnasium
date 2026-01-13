"""Tests of the Taxi env."""

from typing import Final, Literal

import pytest

from gymnasium.envs.toy_text.taxi import Actions, Locations, TaxiEnv


@pytest.mark.parametrize(
    "taxi_row, taxi_col, pass_loc, dest_idx",
    [
        (0, 0, Locations.RED, Locations.GREEN),
        (4, 4, Locations.TAXI, Locations.BLUE),
        (2, 3, Locations.YELLOW, Locations.RED),
        (1, 2, Locations.GREEN, Locations.YELLOW),
    ],
)
def test_taxi_encode_decode_roundtrip(
    taxi_row: int, taxi_col: int, pass_loc: Locations, dest_idx: Locations
) -> None:
    """Test that state -> encode -> decode -> state is correct."""
    state = TaxiEnv.encode(taxi_row, taxi_col, pass_loc, dest_idx)
    decoded = TaxiEnv.decode(state)
    assert decoded == (taxi_row, taxi_col, pass_loc, dest_idx)


@pytest.mark.parametrize(
    "parameters, encoded",
    [
        ((0, 0, Locations.RED, Locations.GREEN), 1),
        ((4, 4, Locations.TAXI, Locations.BLUE), 499),
        ((2, 3, Locations.YELLOW, Locations.RED), 268),
    ],
)
def test_taxi_encode_known_values(
    parameters: tuple[int, int, Locations, Locations],
    encoded: int,
) -> None:
    """Test that encoding of known values is correct."""
    state = TaxiEnv.encode(*parameters)
    assert state == encoded


@pytest.mark.parametrize(
    "taxi_loc, pass_idx, expected_new_pass_idx, expected_reward",
    [
        # Legal pickup: passenger at taxi location, not in taxi
        ((0, 0), Locations.RED, Locations.TAXI, TaxiEnv.PENALTY_STEP),
        ((0, 4), Locations.GREEN, Locations.TAXI, TaxiEnv.PENALTY_STEP),
        # Illegal pickup: passenger not at taxi location
        (
            (0, 0),
            Locations.GREEN,
            Locations.GREEN,
            TaxiEnv.PENALTY_ILLEGAL_PICKUP_DROPOFF,
        ),
        (
            (4, 3),
            Locations.YELLOW,
            Locations.YELLOW,
            TaxiEnv.PENALTY_ILLEGAL_PICKUP_DROPOFF,
        ),
        # Illegal pickup: passenger already in taxi
        (
            (2, 2),
            Locations.TAXI,
            Locations.TAXI,
            TaxiEnv.PENALTY_ILLEGAL_PICKUP_DROPOFF,
        ),
    ],
)
def test_taxi_pickup(
    taxi_loc: tuple[int, int],
    pass_idx: Locations,
    expected_new_pass_idx: Locations,
    expected_reward: int,
) -> None:
    """Test that pickup transitions are correct."""
    env = TaxiEnv()
    new_pass_idx, reward = env._pickup(taxi_loc, pass_idx)  # noqa: SLF001
    assert new_pass_idx == expected_new_pass_idx
    assert reward == expected_reward


@pytest.mark.parametrize(
    "taxi_loc, pass_idx, dest_idx, expected_new_pass_idx, expected_reward, expected_terminated",
    [
        # Legal dropoff: taxi at destination, passenger in taxi
        (
            (0, 0),
            Locations.TAXI,
            Locations.RED,
            Locations.RED,
            TaxiEnv.REWARD_COMPLETE,
            True,
        ),
        (
            (0, 4),
            Locations.TAXI,
            Locations.GREEN,
            Locations.GREEN,
            TaxiEnv.REWARD_COMPLETE,
            True,
        ),
        # Legal dropoff: taxi at valid location, passenger in taxi, but not destination
        (
            (4, 0),
            Locations.TAXI,
            Locations.GREEN,
            Locations.YELLOW,
            TaxiEnv.PENALTY_STEP,
            False,
        ),
        # Illegal dropoff: taxi not at valid location, passenger in taxi
        (
            (2, 2),
            Locations.TAXI,
            Locations.RED,
            Locations.TAXI,
            TaxiEnv.PENALTY_ILLEGAL_PICKUP_DROPOFF,
            False,
        ),
        # Illegal dropoff: passenger not in taxi
        (
            (0, 0),
            Locations.RED,
            Locations.RED,
            Locations.RED,
            TaxiEnv.PENALTY_ILLEGAL_PICKUP_DROPOFF,
            False,
        ),
        (
            (0, 3),
            Locations.YELLOW,
            Locations.GREEN,
            Locations.YELLOW,
            TaxiEnv.PENALTY_ILLEGAL_PICKUP_DROPOFF,
            False,
        ),
    ],
)
def test_taxi_dropoff(  # noqa: PLR0913
    taxi_loc: tuple[int, int],
    pass_idx: Locations,
    dest_idx: Locations,
    expected_new_pass_idx: Locations,
    expected_reward: int,
    expected_terminated: bool,  # noqa: FBT001
) -> None:
    """Test that dropoff transitions are correct."""
    env = TaxiEnv()
    new_pass_idx, reward, terminated = env._dropoff(taxi_loc, pass_idx, dest_idx)  # noqa: SLF001
    assert new_pass_idx == expected_new_pass_idx
    assert reward == expected_reward
    assert terminated == expected_terminated


def test_taxi_transition() -> None:
    """Test that transitions match expectations."""
    prob_idx: Final[Literal[0]] = 0
    state_idx: Final[Literal[1]] = 1
    reward_idx: Final[Literal[2]] = 2
    term_idx: Final[Literal[3]] = 3
    decode = TaxiEnv.decode
    env = None
    for is_rainy in (True, False):
        env = TaxiEnv(is_rainy=is_rainy)
        for state, actions in env.P.items():
            for a, transitions in actions.items():
                action = Actions(a)
                # probabilities should sum to one
                total_prob = sum([t[prob_idx] for t in transitions])
                assert total_prob == pytest.approx(1.0)
                if action == Actions.DROPOFF:
                    for t in transitions:
                        # if ending, should be a success reward
                        if t[term_idx]:
                            assert t[reward_idx] == TaxiEnv.REWARD_COMPLETE
                else:
                    # only dropoff can terminate
                    assert not any(t[term_idx] for t in transitions)
                if action in {Actions.DROPOFF, Actions.PICKUP}:
                    # should be only one transition, always
                    assert len(transitions) == 1
                    # pickup and dropoff should not change taxi location
                    row, col, _, _ = decode(state)
                    new_row, new_col, _, _ = decode(transitions[0][state_idx])
                    assert row == new_row
                    assert col == new_col
                else:
                    # should be only one or three transition
                    assert len(transitions) in [1, 3]
                    # move actions should not change passenger location/destination
                    row, col, loc, dest = decode(state)
                    for t in transitions:
                        new_row, new_col, new_loc, new_dest = decode(t[state_idx])
                        assert loc == new_loc
                        assert dest == new_dest
                        # confirm moves are a single step at most
                        assert abs(row - new_row) + abs(col - new_col) <= 1
    # should be 500 states
    assert env is not None
    assert len(env.P) == 500


@pytest.mark.parametrize(
    "taxi_pos, moves",
    [
        (
            (0, 0),
            {
                # straight ahead, rainy_left, rainy_right
                Actions.MOVE_SOUTH: [(1, 0), (0, 1), (0, 0)],
                Actions.MOVE_NORTH: [(0, 0)],
                Actions.MOVE_EAST: [(0, 1), (0, 0), (1, 0)],
                Actions.MOVE_WEST: [(0, 0)],
            },
        ),
        (
            (4, 3),
            {
                Actions.MOVE_SOUTH: [(4, 3)],
                Actions.MOVE_NORTH: [(3, 3), (4, 3), (4, 4)],
                Actions.MOVE_EAST: [(4, 4), (3, 3), (4, 3)],
                Actions.MOVE_WEST: [(4, 3)],
            },
        ),
        (
            (1, 3),
            {
                Actions.MOVE_SOUTH: [(2, 3), (1, 4), (1, 2)],
                Actions.MOVE_NORTH: [(0, 3), (1, 2), (1, 4)],
                Actions.MOVE_EAST: [(1, 4), (0, 3), (2, 3)],
                Actions.MOVE_WEST: [(1, 2), (2, 3), (0, 3)],
            },
        ),
    ],
)
def test_taxi_specific_move_transions(
    taxi_pos: tuple[int, int], moves: dict[Actions, list[tuple[int, int]]]
) -> None:
    """Test that known transitions are correct."""
    env_dry = TaxiEnv(is_rainy=False)
    env_wet = TaxiEnv(is_rainy=True)

    pass_loc = Locations.RED
    pass_dest = Locations.BLUE
    state = TaxiEnv.encode(*taxi_pos, pass_loc, pass_dest)
    for a in env_dry.P[state]:
        action = Actions(a)
        if action in {Actions.DROPOFF, Actions.PICKUP}:
            continue
        dry_transition = env_dry.P[state][action][0]
        dry_row, dry_col, _, _ = TaxiEnv.decode(dry_transition[1])
        assert (dry_row, dry_col) == moves[action][0]

        wet_transitions = env_wet.P[state][action]
        for expected, transition in zip(moves[action], wet_transitions, strict=True):
            wet_row, wet_col, _, _ = TaxiEnv.decode(transition[1])
            assert (wet_row, wet_col) == expected


def test_taxi_clone_and_restore_state() -> None:
    """Confirm that cloning and restoring state yields correct state."""
    actions = [1, 2, 3, 4]
    for rainy in [True, False]:
        env = TaxiEnv(is_rainy=rainy)
        obs, info = env.reset()
        # Take a few steps to change the state
        for action in actions:
            env.step(action)
        # Clone the current state
        state = env.clone_state()
        # Take more steps to change the state further
        for _ in range(2):
            env.step(env.action_space.sample())
        # Restore the cloned state
        env.restore_state(state)
        # After restoring, the state should match the cloned state
        assert env.s == state.s
        assert env.lastaction == state.lastaction
        assert env.fickle_step == state.fickle_step
        assert env.taxi_orientation == state.taxi_orientation
        # Check that the random state is restored by resetting (to get a new state)
        # then restoring and resetting again to confirm the same initial state
        initial_state_before, _ = env.reset()
        env.restore_state(state)
        initial_state_after, _ = env.reset()
        assert initial_state_before == initial_state_after


@pytest.mark.parametrize(
    "taxi_row, taxi_col, pass_loc, dest_idx, expected_mask",
    [
        # Case 1: Corner position (top-left) - limited movement, no pickup/dropoff
        # Can only move SOUTH and EAST, walls block WEST
        (0, 0, Locations.GREEN, Locations.BLUE, [1, 0, 1, 0, 0, 0]),
        # Case 2: Valid pickup scenario - taxi at passenger location (RED at 0,0)
        # Can move SOUTH and EAST, and can PICKUP
        (0, 0, Locations.RED, Locations.BLUE, [1, 0, 1, 0, 1, 0]),
        # Case 3: Valid dropoff at destination - passenger in taxi at destination
        # Taxi at GREEN (0,4) with passenger in taxi, destination is GREEN
        (0, 4, Locations.TAXI, Locations.GREEN, [1, 0, 0, 1, 0, 1]),
        # Case 4: Valid dropoff at non-destination location - passenger in taxi
        # Taxi at YELLOW (4,0) with passenger, can dropoff at any valid location
        (4, 0, Locations.TAXI, Locations.BLUE, [0, 1, 0, 0, 0, 1]),
        # Case 5: Center position with walls - test wall blocking
        # Taxi at (3,0) next to vertical wall on the right
        (3, 0, Locations.GREEN, Locations.RED, [1, 1, 0, 0, 0, 0]),
    ],
)
def test_taxi_specific_action_mask(
    taxi_row: int,
    taxi_col: int,
    pass_loc: Locations,
    dest_idx: Locations,
    expected_mask: list[int],
) -> None:
    """Test action mask for specific taxi positions and states."""
    env = TaxiEnv()
    state = env.encode(taxi_row, taxi_col, pass_loc, dest_idx)
    mask = env.action_mask(state)

    # Verify the mask matches expected values
    assert list(mask) == expected_mask, (
        f"Action mask mismatch for state (taxi=({taxi_row},{taxi_col}), "
        f"pass_loc={pass_loc}, dest={dest_idx})\n"
        f"Expected: {expected_mask}\n"
        f"Got: {list(mask)}"
    )


def test_taxi_random_action_mask() -> None:
    """Test that action masks for a random state are correct."""
    env = TaxiEnv()

    for state in env.P:
        mask = env.action_mask(state)
        for action, possible in enumerate(mask):
            _, next_state, _, _ = env.P[state][action][0]
            assert state != next_state if possible else state == next_state


def test_taxi_random_encode_decode() -> None:
    """Test that random states correctly encode and decode."""
    env = TaxiEnv()

    state, _ = env.reset()
    for _ in range(100):
        assert env.encode(*env.decode(state)) == state, (
            f"state={state}, encode(decode(state))={env.encode(*env.decode(state))}"
        )
        state, _, _, _, _ = env.step(env.action_space.sample())


def test_taxi_is_rainy() -> None:
    """Test properties of the taxi in a rainy environment."""
    env = TaxiEnv(is_rainy=True)
    for state_dict in env.P.values():
        for action, transitions in state_dict.items():
            if action <= 3:
                assert len(transitions) in [1, 3]
                if len(transitions) == 3:
                    assert sum([t[0] for t in transitions]) == 1
                    assert {t[0] for t in transitions} == {0.8, 0.1}
                if len(transitions) == 1:
                    assert transitions[0][0] == 1.0
            else:
                assert len(transitions) == 1
                assert transitions[0][0] == 1.0

    _ = env.reset()
    _, _, _, _, info = env.step(0)
    assert info["prob"] in {0.8, 0.1, 1.0}

    env = TaxiEnv(is_rainy=False)
    for state_dict in env.P.values():
        for transitions in state_dict.values():
            assert len(transitions) == 1
            assert transitions[0][0] == 1.0

    _ = env.reset()
    _, _, _, _, info = env.step(0)
    assert info["prob"] == 1.0


def test_taxi_disallowed_transitions() -> None:
    """Test that disallowed transitions are not found."""
    disallowed_transitions = [
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
    ]
    for rain in (True, False):
        env = TaxiEnv(is_rainy=rain)
        for state, state_dict in env.P.items():
            start_row, start_col, _, _ = env.decode(state)
            for transitions in state_dict.values():
                for transition in transitions:
                    end_row, end_col, _, _ = env.decode(transition[1])
                    assert (
                        (start_row, start_col),
                        (end_row, end_col),
                    ) not in disallowed_transitions


def test_taxi_fickle_passenger() -> None:
    """Test that the fickle passenger changes their destination."""
    env = TaxiEnv(fickle_passenger=True)

    fickle_seed = 43  # known to produce a fickle step
    env.reset(seed=fickle_seed)
    # If the below assert fails, check if randomness or the draws from the PRNG were
    # recently updated. Alternatively, check if the probability of a fickle step has
    # been updated. If any of these happen, `fickle_seed` may need to be updated to
    # a new seed that triggers the fickle step.
    assert env.fickle_step
    state, *_ = env.step(0)
    _, _, pass_idx, orig_dest_idx = env.decode(state)
    # force taxi to passenger location
    env.s = env.encode(
        env.locs[pass_idx][0],
        env.locs[pass_idx][1],
        pass_idx,
        orig_dest_idx,
    )
    # pick up the passenger
    env.step(4)
    if env.locs[pass_idx][0] == 0:
        # if we're on the top row, move down
        state, *_ = env.step(0)
    else:
        # otherwise move up
        state, *_ = env.step(1)
    _, _, pass_idx, dest_idx = env.decode(state)
    # check that passenger has changed their destination
    assert orig_dest_idx != dest_idx
