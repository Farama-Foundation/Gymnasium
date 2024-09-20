# Gymnasium-docs

This folder contains the documentation for [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

## Instructions for modifying environment pages

### Editing an environment page

Fork Gymnasium and edit the docstring in the environment's Python file. Then, pip install your Gymnasium fork and run `docs/_scripts/gen_mds.py` in this repo. This will automatically generate a Markdown documentation file for the environment.

### Adding a new environment

Ensure the environment is in Gymnasium (or your fork). Ensure that the environment's Python file has a properly formatted markdown docstring. Install using `pip install -e .` and then run `docs/_scripts/gen_mds.py`. This will automatically generate a md page for the environment. Then complete the [other steps](#other-steps).

#### Other steps

- Add the corresponding gif into the `docs/_static/videos/{ENV_TYPE}` folder, where `ENV_TYPE` is the category of your new environment (e.g. mujoco). Follow snake_case naming convention. Alternatively, run `docs/_scripts/gen_gifs.py`.
- Edit `docs/environments/{ENV_TYPE}/index.md`, and add the name of the file corresponding to your new environment to the `toctree`.

## Build the Documentation

Install the required packages and Gymnasium (or your fork):

```
pip install gymnasium
pip install -r docs/requirements.txt
```

To build the documentation once:

```
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml --watch ../gymnasium --re-ignore "pickle$" . _build
```

You can then open http://localhost:8000 in your browser to watch a live updated version of the documentation.

## Writing Tutorials

We use Sphinx-Gallery to build the tutorials inside the `docs/tutorials` directory. Check `docs/tutorials/demo.py` to see an example of a tutorial and [Sphinx-Gallery documentation](https://sphinx-gallery.github.io/stable/syntax.html) for more information.

To convert Jupyter Notebooks to the python tutorials you can use [this script](https://gist.github.com/mgoulao/f07f5f79f6cd9a721db8a34bba0a19a7).

If you want Sphinx-Gallery to execute the tutorial (which adds outputs and plots) then the file name should start with `run_`. Note that this adds to the build time so make sure the script doesn't take more than a few seconds to execute.
