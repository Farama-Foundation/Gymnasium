import os
import sys


all_envs = [
    {
        "id": "mujoco",
        "list": [
            "ant",
            "half_cheetah",
            "hopper",
            "humanoid_standup",
            "humanoid",
            "inverted_double_pendulum",
            "inverted_pendulum",
            "pusher",
            "reacher",
            "swimmer",
            "walker2d",
        ],
    },
    {"id": "toy_text", "list": ["blackjack", "cliff_walking", "frozen_lake", "taxi"]},
    {"id": "box2d", "list": ["bipedal_walker", "car_racing", "lunar_lander"]},
    {
        "id": "classic_control",
        "list": [
            "acrobot",
            "cart_pole",
            "mountain_car_continuous",
            "mountain_car",
            "pendulum",
        ],
    },
]


def create_grid_cell(type_id, env_id, base_path):
    return f"""
            <a href="{base_path}{env_id}">
                <div class="env-grid__cell">
                    <div class="cell__image-container">
                        <img src="/_static/videos/{type_id}/{env_id}.gif">
                    </div>
                    <div class="cell__title">
                        <span>{' '.join(env_id.split('_')).title()}</span>
                    </div>
                </div>
            </a>
    """


def generate_page(env, limit=-1, base_path=""):
    env_type_id = env["id"]
    env_list = env["list"]
    cells = [create_grid_cell(env_type_id, env_id, base_path) for env_id in env_list]
    non_limited_page = limit == -1 or limit >= len(cells)
    if non_limited_page:
        cells = "\n".join(cells)
    else:
        cells = "\n".join(cells[:limit])

    more_btn = (
        """
<a href="./complete_list">
    <button class="more-btn">
        See More Environments
    </button>
</a>
"""
        if not non_limited_page
        else ""
    )
    return f"""
<div class="env-grid">
    {cells}
</div>
{more_btn}
    """


if __name__ == "__main__":
    """
    python gen_envs_display [ env_type ]
    """

    type_dict_arr = []
    type_arg = ""

    if len(sys.argv) > 1:
        type_arg = sys.argv[1]

    for env in all_envs:
        if type_arg == env["id"] or type_arg == "":
            type_dict_arr.append(env)

    for type_dict in type_dict_arr:
        type_id = type_dict["id"]
        envs_path = f"../environments/{type_id}"
        if len(type_dict["list"]) > 20:
            page = generate_page(type_dict, limit=8)
            fp = open(
                os.path.join(os.path.dirname(__file__), envs_path, "list.html"),
                "w",
                encoding="utf-8",
            )
            fp.write(page)
            fp.close()

            page = generate_page(type_dict, base_path="../")
            fp = open(
                os.path.join(
                    os.path.dirname(__file__), envs_path, "complete_list.html"
                ),
                "w",
                encoding="utf-8",
            )
            fp.write(page)
            fp.close()

            fp = open(
                os.path.join(os.path.dirname(__file__), envs_path, "complete_list.md"),
                "w",
                encoding="utf-8",
            )
            env_name = " ".join(type_id.split("_")).title()
            fp.write(
                f"# Complete List - {env_name}\n\n"
                + "```{raw} html\n:file: complete_list.html\n```"
            )
            fp.close()
        else:
            page = generate_page(type_dict)
            fp = open(
                os.path.join(os.path.dirname(__file__), envs_path, "list.html"),
                "w",
                encoding="utf-8",
            )
            fp.write(page)
            fp.close()
