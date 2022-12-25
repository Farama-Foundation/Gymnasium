# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
import os
from typing import Any, Dict

from furo import gen_tutorials

import gymnasium


project = "Gymnasium"
copyright = "2022 Farama Foundation"
author = "Farama Foundation"

# The full version, including alpha/beta/rc tags
release = gymnasium.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "myst_parser",
    "furo.gen_tutorials",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["tutorials/demo.rst"]

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

# Autodoc
autoclass_content = "both"
autodoc_preserve_defaults = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "Gymnasium Documentation"
html_baseurl = "https://gymnasium.farama.org"
html_copy_source = False
html_favicon = "_static/img/favicon.png"
html_theme_options = {
    "light_logo": "img/gymnasium_black.svg",
    "dark_logo": "img/gymnasium_white.svg",
    "gtag": "G-6H9C8TWXZ8",
    "description": "A standard API for reinforcement learning and a diverse set of reference environments (formerly Gym)",
    "image": "img/gymnasium-github.png",
    "versioning": True,
}
html_context: Dict[str, Any] = {}
html_context["conf_py_path"] = "/docs/"
html_context["display_github"] = False
html_context["github_user"] = "Farama-Foundation"
html_context["github_repo"] = "Gymnasium"
html_context["github_version"] = "main"
html_context["slug"] = "gymnasium"

html_static_path = ["_static"]
html_css_files = []

# -- Generate Tutorials -------------------------------------------------

gen_tutorials.generate(
    os.path.dirname(__file__),
    os.path.join(os.path.dirname(__file__), "tutorials"),
)
