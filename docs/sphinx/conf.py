# Configuration file for Sphinx documentation builder.
# MLPY Framework Documentation

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'MLPY Framework'
copyright = '2024, MLPY Team'
author = 'MLPY Development Team'
version = '2.0.0'
release = '2.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx_rtd_theme',
    'nbsphinx',  # Para notebooks Jupyter
    'myst_parser'  # Para Markdown
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': None,
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_static_path = ['_static']
html_css_files = ['custom.css']

# Logo y favicon
html_logo = '_static/mlpy_logo.png'
html_favicon = '_static/favicon.ico'

# -- Options for autodoc ----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for autosummary ------------------------------------------------

autosummary_generate = True

# -- Options for Napoleon (Google/NumPy style docstrings) -------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for todo extension ---------------------------------------------

todo_include_todos = True

# -- Options for intersphinx extension --------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'plotly': ('https://plotly.com/python-api-reference/', None),
}

# -- Custom configuration for MLPY ------------------------------------------

# Metadatos adicionales
html_context = {
    'display_github': True,
    'github_user': 'mlpy-team',
    'github_repo': 'mlpy',
    'github_version': 'main',
    'conf_py_path': '/docs/sphinx/',
}

# Custom CSS y JS
html_css_files = [
    'custom.css',
    'mlpy-theme.css'
]

html_js_files = [
    'custom.js'
]

# Configuración para notebooks
nbsphinx_execute = 'never'  # No ejecutar notebooks al compilar
nbsphinx_allow_errors = True

# Configuración adicional para markdown
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Enlaces automáticos para referencias de API
add_module_names = False