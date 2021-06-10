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
import os
import sys
import seqgra

sys.path.insert(0, os.path.abspath("../../seqgra/"))


# -- Project information -----------------------------------------------------

project = "seqgra"
copyright = "2021, Konstantin Krismer"
author = "Konstantin Krismer"

version = seqgra.__version__
# The full version, including alpha/beta/rc tags
release = seqgra.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = ["sphinx.ext.napoleon",
              "sphinx.ext.autodoc",
              "sphinx.ext.viewcode",
              "sphinx.ext.autosummary",
              "sphinxcontrib.apidoc",
              "sphinxarg.ext"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

apidoc_module_dir = "../../seqgra"
apidoc_output_dir = "."
apidoc_extra_args = ["-f", "-e", "-T",
                     "-d 5",
                     "--implicit-namespaces",
                     "--module-first"]

autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'inherited-members': True,
    'no-special-members': True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

add_module_names = False
modindex_common_prefix = ["seqgra."]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 5
}

html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["custom.css"]
