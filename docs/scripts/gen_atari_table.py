import gymnasium
import tabulate
from tqdm import tqdm


def shortened_repr(lst):
    assert all((isinstance(item, int) for item in lst))
    assert len(set(lst)) == len(lst)
    lst = sorted(lst)

    if lst[-1] - lst[0] == len(lst) - 1 and len(lst) > 3:
        return f"`[{lst[0]}, ..., {lst[-1]}]`"
    elif len(lst) > 3 and lst[-2] - lst[0] == len(lst) - 2:
        return f"`[{lst[0]}, ..., {lst[-2]}, {lst[-1]}]`"
    return f"`{str(lst)}`"


def to_gymnasium_spelling(game):
    parts = game.split("_")
    return "".join([part.capitalize() for part in parts])

atari_envs = [
        "adventure",
        "air_raid",
        "alien",
        "amidar",
        "assault",
        "asterix",
        "asteroids",
        "atlantis",
        "bank_heist",
        "battle_zone",
        "beam_rider",
        "berzerk",
        "bowling",
        "boxing",
        "breakout",
        "carnival",
        "centipede",
        "chopper_command",
        "crazy_climber",
        "defender",
        "demon_attack",
        "double_dunk",
        "elevator_action",
        "enduro",
        "fishing_derby",
        "freeway",
        "frostbite",
        "gopher",
        "gravitar",
        "hero",
        "ice_hockey",
        "jamesbond",
        "journey_escape",
        "kangaroo",
        "krull",
        "kung_fu_master",
        "montezuma_revenge",
        "ms_pacman",
        "name_this_game",
        "phoenix",
        "pitfall",
        "pong",
        "pooyan",
        "private_eye",
        "qbert",
        "riverraid",
        "road_runner",
        "robotank",
        "seaquest",
        "skiing",
        "solaris",
        "space_invaders",
        "star_gunner",
        "tennis",
        "time_pilot",
        "tutankham",
        "up_n_down",
        "venture",
        "video_pinball",
        "wizard_of_wor",
        "yars_revenge",
        "zaxxon",
        ]


header = ["Environment", "Valid Modes", "Valid Difficulties", "Default Mode"]
rows = []

for game in tqdm(atari_envs):
    env = gymnasium.make(f"ALE/{to_gymnasium_spelling(game)}-v5")
    valid_modes = env.unwrapped.ale.getAvailableModes()
    valid_difficulties = env.unwrapped.ale.getAvailableDifficulties()
    difficulty = env.unwrapped.ale.cloneState().getDifficulty()
    assert (difficulty == 0), difficulty
    rows.append([to_gymnasium_spelling(game), shortened_repr(valid_modes), shortened_repr(valid_difficulties), f"`{env.unwrapped.ale.cloneState().getCurrentMode()}`"])


print(tabulate.tabulate(rows, headers=header, tablefmt="github"))
