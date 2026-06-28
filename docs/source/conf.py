"""Sphinx configuration for LazyML documentation."""

import os
import sys
from pathlib import Path

# Add the src directory to the path so Sphinx can find the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Project information
project = "LazyML"
copyright = "2024, LazyML Contributors"
author = "LazyML Contributors"
version = "0.1.0"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source parsers
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# HTML output options
html_theme = "furo"
html_static_path = ["_static"]
html_title = "LazyML Documentation"
html_short_title = "LazyML"
html_css_files = ["custom.css"]

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Napoleon settings (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
]

# Language
language = "en"
