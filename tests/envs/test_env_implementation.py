import numpy as np
import pytest

import gymnasium as gym
from gymnasium.envs.box2d import BipedalWalker, CarRacing
from gymnasium.envs.box2d.lunar_lander import demo_heuristic_lander
from gymnasium.envs.toy_text import CliffWalkingEnv, TaxiEnv
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.error import InvalidAction


def test_lunar_lander_heuristics():
    """Tests the LunarLander environment by checking if the heuristic lander works."""
    lunar_lander = gym.make("LunarLander-v3", disable_env_checker=True)
    total_reward = demo_heuristic_lander(lunar_lander, seed=1)
    assert total_reward > 100


@pytest.mark.parametrize("seed", [0, 10, 20, 30, 40])
def test_lunar_lander_random_wind_seed(seed: int):
    """Test that the wind_idx and torque are correctly drawn when setting a seed"""

    lunar_lander = gym.make(
        "LunarLander-v3", disable_env_checker=True, enable_wind=True
    ).unwrapped
    lunar_lander.reset(seed=seed)

    # Test that same seed gives same wind
    w1, t1 = lunar_lander.wind_idx, lunar_lander.torque_idx
    lunar_lander.reset(seed=seed)
    w2, t2 = lunar_lander.wind_idx, lunar_lander.torque_idx
    assert (
        w1 == w2 and t1 == t2
    ), "Setting same seed caused different initial wind or torque index"

    # Test that different seed gives different wind
    # There is a small chance that different seeds causes same number so test
    # 10 times (with different seeds) to make this chance incredibly tiny.
    for i in range(1, 11):
        lunar_lander.reset(seed=seed + i)
        w3, t3 = lunar_lander.wind_idx, lunar_lander.torque_idx
        if w2 != w3 and t1 != t3:  # Found different initial values
            break
    else:  # no break
        raise AssertionError(
            "Setting different seed caused same initial wind or torque index"
        )


def test_carracing_domain_randomize():
    """Tests the CarRacing Environment domain randomization.

    CarRacing DomainRandomize should have different colours at every reset.
    However, it should have same colours when `options={"randomize": False}` is given to reset.
    """
    env: CarRacing = gym.make("CarRacing-v3", domain_randomize=True).unwrapped

    road_color = env.road_color
    bg_color = env.bg_color
    grass_color = env.grass_color

    env.reset(options={"randomize": False})

    assert (
        road_color == env.road_color
    ).all(), f"Have different road color after reset with randomize turned off. Before: {road_color}, after: {env.road_color}."
    assert (
        bg_color == env.bg_color
    ).all(), f"Have different bg color after reset with randomize turned off. Before: {bg_color}, after: {env.bg_color}."
    assert (
        grass_color == env.grass_color
    ).all(), f"Have different grass color after reset with randomize turned off. Before: {grass_color}, after: {env.grass_color}."

    env.reset()

    assert (
        road_color != env.road_color
    ).all(), f"Have same road color after reset. Before: {road_color}, after: {env.road_color}."
    assert (
        bg_color != env.bg_color
    ).all(), (
        f"Have same bg color after reset. Before: {bg_color}, after: {env.bg_color}."
    )
    assert (
        grass_color != env.grass_color
    ).all(), f"Have same grass color after reset. Before: {grass_color}, after: {env.grass_color}."


def test_slippery_cliffwalking():
    """Test that the slippery cliffwalking environment is correctly implemented.
    We check here that there are always 3 possible transitions for each action and
    that there is a 1/3 probability for each.
    """
    envs = CliffWalkingEnv(is_slippery=True)
    for actions_dict in envs.P.values():
        for transitions in actions_dict.values():
            assert len(transitions) == 3
            assert all([r[0] == 1 / 3 for r in transitions])


def test_cliffwalking():
    env = CliffWalkingEnv(is_slippery=False)
    for actions_dict in env.P.values():
        for transitions in actions_dict.values():
            assert len(transitions) == 1
            assert all([r[0] == 1.0 for r in transitions])


@pytest.mark.parametrize("seed", range(5))
def test_bipedal_walker_hardcore_creation(seed: int):
    """Test BipedalWalker hardcore creation.

    BipedalWalker with `hardcore=True` should have ladders
    stumps and pitfalls. A convenient way to identify if ladders,
    stumps and pitfall are created is checking whether the terrain
    has that particular terrain color.

    Args:
        seed (int): environment seed
    """
    HC_TERRAINS_COLOR1 = (255, 255, 255)
    HC_TERRAINS_COLOR2 = (153, 153, 153)

    env = gym.make("BipedalWalker-v3", disable_env_checker=True).unwrapped
    hc_env = gym.make("BipedalWalkerHardcore-v3", disable_env_checker=True).unwrapped
    assert isinstance(env, BipedalWalker) and isinstance(hc_env, BipedalWalker)
    assert env.hardcore is False and hc_env.hardcore is True

    env.reset(seed=seed)
    hc_env.reset(seed=seed)

    for terrain in env.terrain:
        assert terrain.color1 != HC_TERRAINS_COLOR1
        assert terrain.color2 != HC_TERRAINS_COLOR2

    hc_terrains_color1_count = 0
    hc_terrains_color2_count = 0
    for terrain in hc_env.terrain:
        if terrain.color1 == HC_TERRAINS_COLOR1:
            hc_terrains_color1_count += 1
        if terrain.color2 == HC_TERRAINS_COLOR2:
            hc_terrains_color2_count += 1

    assert hc_terrains_color1_count > 0
    assert hc_terrains_color2_count > 0


@pytest.mark.parametrize("map_size", [5, 10, 16])
def test_frozenlake_dfs_map_generation(map_size: int):
    """Frozenlake has the ability to generate random maps.

    This function checks that the random maps will always be possible to solve for sizes 5, 10, 16,
    currently only 8x8 maps can be generated.
    """
    new_frozenlake = generate_random_map(map_size)
    assert len(new_frozenlake) == map_size
    assert len(new_frozenlake[0]) == map_size

    # Runs a depth first search through the map to find the path.
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        row, col = frontier.pop()
        if (row, col) not in discovered:
            discovered.add((row, col))

            for row_direction, col_direction in directions:
                new_row = row + row_direction
                new_col = col + col_direction
                if 0 <= new_row < map_size and 0 <= new_col < map_size:
                    if new_frozenlake[new_row][new_col] == "G":
                        return  # Successful, a route through the map was found
                    if new_frozenlake[new_row][new_col] not in "#H":
                        frontier.append((new_row, new_col))
    raise AssertionError("No path through the frozenlake was found.")


@pytest.mark.parametrize("map_size, seed", [(5, 123), (10, 42), (16, 987)])
def test_frozenlake_map_generation_with_seed(map_size: int, seed: int):
    map1 = generate_random_map(size=map_size, seed=seed)
    map2 = generate_random_map(size=map_size, seed=seed)
    assert map1 == map2
    map1 = generate_random_map(size=map_size, seed=seed)
    map2 = generate_random_map(size=map_size, seed=seed + 1)
    assert map1 != map2


def test_taxi_action_mask():
    env = TaxiEnv()

    for state in env.P:
        mask = env.action_mask(state)
        for action, possible in enumerate(mask):
            _, next_state, _, _ = env.P[state][action][0]
            assert state != next_state if possible else state == next_state


def test_taxi_encode_decode():
    env = TaxiEnv()

    state, info = env.reset()
    for _ in range(100):
        assert (
            env.encode(*env.decode(state)) == state
        ), f"state={state}, encode(decode(state))={env.encode(*env.decode(state))}"
        state, _, _, _, _ = env.step(env.action_space.sample())


def test_taxi_is_rainy():
    env = TaxiEnv(is_rainy=True)
    for state_dict in env.P.values():
        for action, transitions in state_dict.items():
            if action <= 3:
                assert sum([t[0] for t in transitions]) == 1
                assert {t[0] for t in transitions} == {0.8, 0.1}
            else:
                assert len(transitions) == 1
                assert transitions[0][0] == 1.0

    state, _ = env.reset()
    _, _, _, _, info = env.step(0)
    assert info["prob"] in {0.8, 0.1}

    env = TaxiEnv(is_rainy=False)
    for state_dict in env.P.values():
        for action, transitions in state_dict.items():
            assert len(transitions) == 1
            assert transitions[0][0] == 1.0

    state, _ = env.reset()
    _, _, _, _, info = env.step(0)
    assert info["prob"] == 1.0


def test_taxi_disallowed_transitions():
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
    for rain in {True, False}:
        env = TaxiEnv(is_rainy=rain)
        for state, state_dict in env.P.items():
            start_row, start_col, _, _ = env.decode(state)
            for action, transitions in state_dict.items():
                for transition in transitions:
                    end_row, end_col, _, _ = env.decode(transition[1])
                    assert (
                        (start_row, start_col),
                        (end_row, end_col),
                    ) not in disallowed_transitions


def test_taxi_fickle_passenger():
    env = TaxiEnv(fickle_passenger=True)
    # This is a fickle seed, if randomness or the draws from the PRNG were recently updated, find a new seed
    env.reset(seed=43)
    state, *_ = env.step(0)
    taxi_row, taxi_col, pass_idx, orig_dest_idx = env.decode(state)
    # force taxi to passenger location
    env.s = env.encode(
        env.locs[pass_idx][0], env.locs[pass_idx][1], pass_idx, orig_dest_idx
    )
    # pick up the passenger
    env.step(4)
    if env.locs[pass_idx][0] == 0:
        # if we're on the top row, move down
        state, *_ = env.step(0)
    else:
        # otherwise move up
        state, *_ = env.step(1)
    taxi_row, taxi_col, pass_idx, dest_idx = env.decode(state)
    # check that passenger has changed their destination
    assert orig_dest_idx != dest_idx


@pytest.mark.parametrize(
    "env_name",
    ["Acrobot-v1", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0"],
)
@pytest.mark.parametrize(
    "low_high", [None, (-0.4, 0.4), (np.array(-0.4), np.array(0.4))]
)
def test_customizable_resets(env_name: str, low_high: list | None):
    env = gym.make(env_name)
    env.action_space.seed(0)
    # First ensure we can do a reset.
    if low_high is None:
        env.reset()
    else:
        low, high = low_high
        env.reset(options={"low": low, "high": high})
        assert np.all((env.unwrapped.state >= low) & (env.unwrapped.state <= high))
    # Make sure we can take a step.
    env.step(env.action_space.sample())


# We test Pendulum separately, as the parameters are handled differently.
@pytest.mark.parametrize(
    "low_high",
    [
        None,
        (1.2, 1.0),
        (np.array(1.2), np.array(1.0)),
    ],
)
def test_customizable_pendulum_resets(low_high: list | None):
    env = gym.make("Pendulum-v1")
    env.action_space.seed(0)
    # First ensure we can do a reset and the values are within expected ranges.
    if low_high is None:
        env.reset()
    else:
        low, high = low_high
        # Pendulum is initialized a little differently than the other
        # environments, where we specify the x and y values for the upper
        # limit (and lower limit is just the negative of it).
        env.reset(options={"x_init": low, "y_init": high})
    # Make sure we can take a step.
    env.step(env.action_space.sample())


@pytest.mark.parametrize(
    "env_name",
    ["Acrobot-v1", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0"],
)
@pytest.mark.parametrize(
    "low_high",
    [
        ("x", "y"),
        (10.0, 8.0),
        ([-1.0, -1.0], [1.0, 1.0]),
        (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
    ],
)
def test_invalid_customizable_resets(env_name: str, low_high: list):
    env = gym.make(env_name)
    low, high = low_high
    with pytest.raises(ValueError):
        # match=re.escape(f"Lower bound ({low}) must be lower than higher bound ({high}).")
        # match=f"An option ({x}) could not be converted to a float."
        env.reset(options={"low": low, "high": high})


def test_cartpole_vector_equiv():
    env = gym.make("CartPole-v1")
    envs = gym.make_vec("CartPole-v1", num_envs=1)

    assert env.action_space == envs.single_action_space
    assert env.observation_space == envs.single_observation_space

    # for seed in range(0, 10_000):
    seed = np.random.randint(0, 1000)

    # reset
    obs, info = env.reset(seed=seed)
    vec_obs, vec_info = envs.reset(seed=seed)

    env.action_space.seed(seed=seed)

    assert obs in env.observation_space
    assert vec_obs in envs.observation_space
    assert np.all(obs == vec_obs[0])
    assert info == vec_info

    assert np.all(env.unwrapped.state == envs.unwrapped.state[:, 0])

    # step
    for i in range(100):
        action = env.action_space.sample()
        assert np.array([action]) in envs.action_space

        obs, reward, term, trunc, info = env.step(action)
        vec_obs, vec_reward, vec_term, vec_trunc, vec_info = envs.step(
            np.array([action])
        )

        assert obs in env.observation_space
        assert vec_obs in envs.observation_space
        assert np.all(obs == vec_obs[0])
        assert reward == vec_reward
        assert term == vec_term
        assert trunc == vec_trunc
        assert info == vec_info

        assert np.all(env.unwrapped.state == envs.unwrapped.state[:, 0])

        if term or trunc:
            break

    # if the sub-environment episode ended
    if term or trunc:
        obs, info = env.reset()
        # the vector action shouldn't matter as autoreset
        assert envs.unwrapped.prev_done
        vec_obs, vec_reward, vec_term, vec_trunc, vec_info = envs.step(
            envs.action_space.sample()
        )

        assert obs in env.observation_space
        assert vec_obs in envs.observation_space
        assert np.all(obs == vec_obs[0])
        assert vec_reward == np.array([0])
        assert vec_term == np.array([False])
        assert vec_trunc == np.array([False])
        assert info == vec_info

        assert np.all(env.unwrapped.state == envs.unwrapped.state[:, 0])

    env.close()
    envs.close()


@pytest.mark.parametrize("env_id", ["CarRacing-v3", "LunarLander-v3"])
def test_discrete_action_validation(env_id):
    # get continuous action
    continuous_env = gym.make(env_id, continuous=True)
    continuous_action = continuous_env.action_space.sample()
    continuous_env.close()

    # create discrete env
    discrete_env = gym.make(env_id, continuous=False)
    discrete_env.reset()

    # expect InvalidAction (caused by CarRacing) or AssertionError (caused by LunarLander)
    with pytest.raises((InvalidAction, AssertionError)):
        discrete_env.step(continuous_action)

    # expect no error
    discrete_action = discrete_env.action_space.sample()
    discrete_env.step(discrete_action)
    discrete_env.close()
