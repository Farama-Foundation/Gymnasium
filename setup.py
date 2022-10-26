"""Setups the project."""
import itertools

from setuptools import find_packages, setup


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as file:
        long_description = ""
        header_count = 0
        for line in file:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return header_count, long_description


def get_version():
    """Gets the gymnasium version."""
    path = "gymnasium/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


# Environment-specific dependencies.
extras = {
    "atari": ["ale-py~=0.8.0"],
    "accept-rom-license": ["autorom[accept-rom-license]~=0.4.2"],
    "box2d": ["box2d-py==2.3.5", "pygame==2.1.0", "swig==4.*"],
    "classic_control": ["pygame==2.1.0"],
    "mujoco_py": ["mujoco_py<2.2,>=2.1"],
    "mujoco": ["mujoco==2.2", "imageio>=2.14.1"],
    "toy_text": ["pygame==2.1.0"],
    "other": ["lz4>=3.1.0", "opencv-python>=3.0", "matplotlib>=3.0", "moviepy>=1.0.0"],
}

# Testing dependency groups.
testing_group = set(extras.keys()) - {"accept-rom-license", "atari"}
extras["testing"] = list(
    set(itertools.chain.from_iterable(map(lambda group: extras[group], testing_group)))
) + ["pytest==7.0.1", "gym==0.26.2"]

# All dependency groups - accept rom license as requires user to run
all_groups = set(extras.keys()) - {"accept-rom-license"}
extras["all"] = list(
    set(itertools.chain.from_iterable(map(lambda group: extras[group], all_groups)))
)

version = get_version()
header_count, long_description = get_description()

setup(
    name="Gymnasium",
    version=version,
    author="Farama Foundation",
    author_email="contact@farama.org",
    description="A standard API for reinforcement learning and a diverse set of reference environments (formerly Gym)",
    url="https://gymnasium.farama.org/",
    license="MIT",
    license_files=("LICENSE",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Reinforcement Learning", "game", "RL", "AI", "gymnasium"],
    python_requires=">=3.6",
    tests_require=extras["testing"],
    packages=[
        package for package in find_packages() if package.startswith("gymnasium")
    ],
    package_data={
        "gymnasium": [
            "envs/mujoco/assets/*.xml",
            "envs/classic_control/assets/*.png",
            "envs/toy_text/font/*.ttf",
            "envs/toy_text/img/*.png",
            "py.typed",
        ]
    },
    include_package_data=True,
    install_requires=[
        "numpy >= 1.18.0",
        "cloudpickle >= 1.2.0",
        "importlib_metadata >= 4.8.0; python_version < '3.10'",
        "gymnasium_notices >= 0.0.1",
        "dataclasses == 0.8; python_version == '3.6'",
    ],
    classifiers=[
        # Python 3.6 is minimally supported (only with basic gymnasium environments and API)
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    extras_require=extras,
    zip_safe=False,
)
