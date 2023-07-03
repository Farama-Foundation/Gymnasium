import itertools
import json

import tabulate
from ale_py.roms import utils as rom_utils
from shimmy.utils.envs_configs import ALL_ATARI_GAMES
from tqdm import tqdm

import gymnasium


# # Generate the list of all atari games on atari.md
for rom_id in sorted(ALL_ATARI_GAMES):
    print(f"atari/{rom_id}")


def generate_value_ranges(values):
    for a, b in itertools.groupby(enumerate(values), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1], b[-1][1]


def shortened_repr(values):
    output = []
    for low, high in generate_value_ranges(values):
        if high - low < 5:
            output.append(", ".join(map(str, range(low, high + 1))))
        else:
            output.append(f"{low}, ..., {high}")
    return "[" + ", ".join(output) + "]"


# # Test examples
# print(shortened_repr([0]))
# print(shortened_repr([1, 2, 3]))
# print(shortened_repr([0, 1, 2, 3]))
# print(shortened_repr([0, 4, 8, 12, 16, 20, 24, 28]))
# print(shortened_repr(list(range(32)) + [128]))


# # Generate difficult levels table on atari.md
headers = [
    "Environment",
    "Possible Modes",
    "Default Mode",
    "Possible Difficulties",
    "Default Difficulty",
]
rows = []

for rom_id in tqdm(ALL_ATARI_GAMES):
    env_name = rom_utils.rom_id_to_name(rom_id)

    env = gymnasium.make(f"ALE/{env_name}-v5")

    available_difficulties = env.ale.getAvailableDifficulties()
    default_difficulty = env.ale.cloneState().getDifficulty()
    available_modes = env.ale.getAvailableModes()
    default_mode = env.ale.cloneState().getCurrentMode()

    if env_name == "VideoCube":
        available_modes = "[0, 1, 2, 100, 101, 102, ..., 5000, 5001, 5002]"
    else:
        available_modes = shortened_repr(available_modes)

    rows.append(
        [
            env_name,
            available_modes,
            default_mode,
            shortened_repr(available_difficulties),
            default_difficulty,
        ]
    )
    env.close()

print(tabulate.tabulate(rows, headers=headers, tablefmt="github"))

# Generate each pages results
with open("atari-docs.json") as file:
    atari_data = json.load(file)

for rom_id in tqdm(ALL_ATARI_GAMES):
    env_name = rom_utils.rom_id_to_name(rom_id)

    env = gymnasium.make(f"ALE/{env_name}-v5")
    if rom_id in atari_data:
        env_data = atari_data[rom_id]

        env_description = env_data["env_description"]
        if env_data["atariage_url"]:
            env_url = f"""
For a more detailed documentation, see [the AtariAge page]({env_data['atariage_url']})
"""
        else:
            env_url = ""
        reward_description = env_data["reward_description"]
    else:
        # Add the information to `atari_docs.json` and rerun this file to generate the new documentation
        env_description = f"{env_name} is missing description documentation. If you are interested in writing up a description, please create an issue or PR with the information on the Gymnasium github."
        env_url = ""
        reward_description = ""

    table_values = map(
        lambda s: f"`{s}`",
        itertools.chain(*zip(range(env.action_space.n), env.get_action_meanings())),
    )
    default_action_table = tabulate.tabulate(
        list(itertools.zip_longest(*([iter(table_values)] * 6), fillvalue="")),
        headers=["Value", "Meaning", "Value", "Meaning", "Value", "Meaning"],
        tablefmt="github",
    )
    if env.action_space.n == 18:
        action_description = f"""{env_name} has the action space `{env.action_space}` with the table below listing the meaning of each action's meanings.
As {env_name} uses the full set of actions then specifying `full_action_space=True` will not modify the action space of the environment if passed to `gymnasium.make`."""
    else:
        action_description = f"""{env_name} has the action space of `{env.action_space}` with the table below listing the meaning of each action's meanings.
To enable all 18 possible actions that can be performed on an Atari 2600, specify `full_action_space=True` during
initialization or by passing `full_action_space=True` to `gymnasium.make`."""

    # Environment variants
    env_specs = sorted(
        [
            env_spec
            for env_spec in gymnasium.registry.values()
            if env_name in env_spec.name and "shimmy" in env_spec.entry_point
        ],
        key=lambda env_spec: f"{env_spec.version}{env_spec.name}",
    )

    env_variant_headers = [
        "Env-id",
        "obs_type=",
        "frameskip=",
        "repeat_action_probability=",
    ]
    env_variant_rows = [
        [
            env_spec.id,
            f'`"{env_spec.kwargs["obs_type"]}"`',
            f'`{env_spec.kwargs["frameskip"]}`',
            f'`{env_spec.kwargs["repeat_action_probability"]}`',
        ]
        for env_spec in env_specs
    ]
    env_variant_table = tabulate.tabulate(
        env_variant_rows, headers=env_variant_headers, tablefmt="github"
    )

    # difficult and mode description

    difficulty_mode_header = [
        "Available Modes",
        "Default Mode",
        "Available Difficulties",
        "Default Difficulty",
    ]
    difficulty_mode_row = [
        [
            f"`{shortened_repr(env.ale.getAvailableModes())}`",
            f"`{env.ale.cloneState().getCurrentMode()}`",
            f"`{shortened_repr(env.ale.getAvailableDifficulties())}`",
            f"`{env.ale.cloneState().getDifficulty()}`",
        ]
    ]
    difficulty_mode_table = tabulate.tabulate(
        difficulty_mode_row, headers=difficulty_mode_header, tablefmt="github"
    )

    env.close()

    TEMPLATE = f"""---
title: {env_name}
---

# {env_name}

```{{figure}} ../../_static/videos/atari/{rom_id}.gif
:width: 120px
:name: {env_name}
```

This environment is part of the <a href='..'>Atari environments</a>. Please read that page first for general information.

|   |   |
|---|---|
| Action Space | {env.action_space} |
| Observation Space | {env.observation_space} |
| Import | `gymnasium.make("{env.spec.id}")` |

For more {env_name} variants with different observation and action spaces, see the variants section.

## Description

{env_description}
{env_url}
## Actions

{action_description}

{default_action_table}

## Observations

Atari environments have three possible observation types: `"rgb"`, `"grayscale"` and `"ram"`.

- `obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)`
- `obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)`
- `obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8)`, a grayscale version of the "rgb" type

See variants section for the type of observation used by each environment id by default.

{reward_description}
## Variants

{env_name} has the following variants of the environment id which have the following differences in observation,
the number of frame-skips and the repeat action probability.

{env_variant_table}

## Difficulty and modes

It is possible to specify various flavors of the environment via the keyword arguments `difficulty` and `mode`.
A flavor is a combination of a game mode and a difficulty setting. The table below lists the possible difficulty and mode values
along with the default values.

{difficulty_mode_table}

## Version History

A thorough discussion of the intricate differences between the versions and configurations can be found in the general article on Atari environments.

* v5: Stickiness was added back and stochastic frameskipping was removed. The environments are now in the "ALE" namespace.
* v4: Stickiness of actions was removed
* v0: Initial versions release
"""
    with open(f"../environments/atari/{rom_id}.md", "w") as file:
        file.write(TEMPLATE)
