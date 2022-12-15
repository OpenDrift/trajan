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

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')
sphinx_gallery_conf = {
     'examples_dirs': '../../examples/',   # path to your example scripts
     'gallery_dirs': './gallery',  # path to where to save gallery generated output,
     'filename_pattern': '/example_(?!long_)',
     'ignore_pattern': 'create_test',
     'backreferences_dir': None,
     'capture_repr': ('_repr_html_', '__repr__'),
     'abort_on_example_error': False,
     'thumbnail_size': (300, 300),
     'junit': '../../test-results/sphinx-gallery/junit.xml',
}

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
    'matplotlib.sphinxext.plot_directive',
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
