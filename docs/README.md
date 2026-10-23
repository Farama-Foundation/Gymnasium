# Gymnasium-docs

This folder contains the documentation for [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

## Instructions for modifying environment pages

### Editing an environment page

Fork Gymnasium and edit the docstring in the environment's Python file. Then, pip install your Gymnasium fork and run `docs/_scripts/gen_mds.py` in this repo. This will automatically generate a Markdown documentation file for the environment.

### Adding a new environment

Ensure the environment is in Gymnasium (or your fork). Ensure that the environment's Python file has a properly formatted markdown docstring. Install using `pip install -e .` and then run `docs/_scripts/gen_mds.py`. This will automatically generate a md page for the environment. Then complete the [other steps](#other-steps).

#### Other steps

- Add the corresponding gif into the `docs/_static/videos/{ENV_TYPE}` folder, where `ENV_TYPE` is the category of your new environment (e.g. mujoco). Follow snake_case naming convention. Alternatively, run `docs/_scripts/gen_gifs.py`.
- Edit `docs/environments/{ENV_TYPE}.md`, and add the name of the file corresponding to your new environment to the `toctree`.

## Build the Documentation

Install the required packages and Gymnasium (or your fork):

```
pip install -e .
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

We use Sphinx-Gallery to build the tutorials inside the `docs/tutorials` directory. Tutorials live in `docs/tutorials/gymnasium_basics` or `docs/tutorials/training_agents`. See `docs/tutorials/gymnasium_basics/environment_creation.py` for an example of a tutorial and the [Sphinx-Gallery documentation](https://sphinx-gallery.github.io/stable/syntax.html) for more information.

To convert Jupyter Notebooks to the python tutorials you can use [this script](https://gist.github.com/mgoulao/f07f5f79f6cd9a721db8a34bba0a19a7).

Tutorials are not executed during the documentation build (the `filename_pattern` option in `docs/conf.py` is disabled), so the rendered pages show the code without outputs or plots. Instead, the `Run Tutorials` workflow (`.github/workflows/run-tutorial.yml`) runs each modified tutorial on pull requests, with fewer training iterations.

## Testing the Documentation

To prevent the documentation from drifting from the code, `tests/test_docs_code.py` (run with the rest of the tests, `pytest tests/test_docs_code.py`) checks that:

- The ```` ```python ```` code blocks of the pages in `docs/introduction`, `docs/api` and `docs/environments/mujoco.md` run. Each page's blocks are run in order with a shared namespace, and blocks containing `>>>` are run as doctests, so their outputs must match. To skip a block that is not meant to be run (e.g., the old Gym API or a code fragment), add `<!-- test: skip -->` on the line before it.
- `docs/api/wrappers/table.md` is up to date with the wrappers, run `python docs/_scripts/gen_wrapper_table.py` to update it after adding or changing a wrapper.

The docstring examples in `gymnasium/` are tested with `pytest --doctest-modules gymnasium/`.
