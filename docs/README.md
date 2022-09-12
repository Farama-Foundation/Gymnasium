# Gymnasium-docs

This folder contains the documentation for [Gymnasium](https://github.com/Farama-Foundation/gymnasium).

If you are modifying a non-environment page or an atari environment page, please PR this repo. Otherwise, follow the steps below:

## Instructions for modifying environment pages

### Editing an environment page

If you are editing an Atari environment, directly edit the Markdown file in this repository. 

Otherwise, fork Gymnasium and edit the docstring in the environment's Python file. Then, pip install your Gymnasium fork and run `docs/scripts/gen_mds.py` in this repo. This will automatically generate a Markdown documentation file for the environment.

### Adding a new environment

#### Atari env

For Atari envs, add a Markdown file into `pages/environments/atari` then complete the **other steps**.

#### Non-Atari env

Ensure the environment is in Gymnasium (or your fork). Ensure that the environment's Python file has a properly formatted markdown docstring. Pip install Gymnasium (or your fork) then run `docs/scripts/gen_mds.py`. This will automatically generate an md page for the environment. Then complete the [other steps](#other-steps).

#### Other steps

- Add the corresponding gif into the `docs/_static/videos/{ENV_TYPE}` folder, where `ENV_TYPE` is the category of your new environment (e.g. mujoco). Follow snake_case naming convention. Alternatively, run `docs/scripts/gen_gifs.py`.
- Edit `docs/environments/{ENV_TYPE}/index.md`, and add the name of the file corresponding to your new environment to the `toctree`.

## Build the Documentation

Install the required packages and Gymnasium (or your fork):

```
pip install -r requirements.txt
pip install gymnasium
```

To build the documentation once:

```
cd docs
make dirhtml _build
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml . _build
```
