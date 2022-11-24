import os, sys
sys.path.insert(0, os.path.abspath('../..'))
# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Trajan'
copyright = '2022, G. Hope and K. F. Dagestad'
author = 'G. Hope and K. F. Dagestad'

release = '0.1'
version = '0.1.0'

# -- General configuration

autoapi_type = 'python'
autoapi_dirs = [ '../../trajan' ]
autoapi_keep_files = False  # set to True when debugging autoapi generated files
autoapi_python_class_content = 'both'
autodoc_typehints = 'description'

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
