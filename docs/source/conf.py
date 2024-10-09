import os, sys
sys.path.insert(0, os.path.abspath('../..'))

import trajan  # noqa

import sphinx_autosummary_accessors
# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'TrajAn'
copyright = '2022-2024, G. Hope and K. F. Dagestad'
author = 'G. Hope and K. F. Dagestad'

release = 'latest (git / main)'
version = 'latest (git / main)'

# -- General configuration

autoapi_type = 'python'
autoapi_dirs = [ '../../trajan' ]
autoapi_keep_files = False  # set to True when debugging autoapi generated files
autoapi_python_class_content = 'both'
autodoc_typehints = 'none'

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
    #"sphinxcontrib.mermaid",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    #'autoapi.extension',
    'sphinx.ext.autosummary',
    'sphinx_autosummary_accessors',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
    #'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'cftime': ('https://unidata.github.io/cftime', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'pyproj': ('https://pyproj4.github.io/pyproj/stable/', None),
    'xarray': ('https://xarray.pydata.org/en/stable/', None),
    'pandas': ("https://pandas.pydata.org/pandas-docs/stable", None),
}
#intersphinx_disabled_domains = ['std']

templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "OpenDrift",
    "github_repo": "trajan",
    "github_version": "main",
    "doc_path": "docs",
}

html_theme_options = {
    "use_edit_page_button": True,
}

autosummary_generate = True
autodoc_typehints = "none"
autodoc_typehints_description_target = "documented"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    "file-like": ":term:`file-like <file-like object>`",
    # special terms
    # "same type as caller": "*same type as caller*",  # does not work, yet
    # "same type as values": "*same type as values*",  # does not work, yet
    # stdlib type aliases
    "MutableMapping": "~collections.abc.MutableMapping",
    "sys.stdout": ":obj:`sys.stdout`",
    "timedelta": "~datetime.timedelta",
    "string": ":class:`string <str>`",
    # numpy terms
    "array_like": ":term:`array_like`",
    "array-like": ":term:`array-like <array_like>`",
    "scalar": ":term:`scalar`",
    "array": ":term:`array`",
    "hashable": ":term:`hashable <name>`",
    # matplotlib terms
    "color-like": ":py:func:`color-like <matplotlib.colors.is_color_like>`",
    "matplotlib colormap name": ":doc:`matplotlib colormap name <matplotlib:gallery/color/colormap_reference>`",
    "matplotlib axes object": ":py:class:`matplotlib axes object <matplotlib.axes.Axes>`",
    "colormap": ":py:class:`colormap <matplotlib.colors.Colormap>`",
    # xarray terms
    "dim name": ":term:`dimension name <name>`",
    "var name": ":term:`variable name <name>`",
    # objects without namespace: xarray
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    "Variable": "~xarray.Variable",
    "DataTree": "~xarray.DataTree",
    "DatasetGroupBy": "~xarray.core.groupby.DatasetGroupBy",
    "DataArrayGroupBy": "~xarray.core.groupby.DataArrayGroupBy",
    "Grouper": "~xarray.groupers.Grouper",
    "Resampler": "~xarray.groupers.Resampler",
    # objects without namespace: numpy
    "ndarray": "~numpy.ndarray",
    "MaskedArray": "~numpy.ma.MaskedArray",
    "dtype": "~numpy.dtype",
    "ComplexWarning": "~numpy.ComplexWarning",
    # objects without namespace: pandas
    "Index": "~pandas.Index",
    "MultiIndex": "~pandas.MultiIndex",
    "CategoricalIndex": "~pandas.CategoricalIndex",
    "TimedeltaIndex": "~pandas.TimedeltaIndex",
    "DatetimeIndex": "~pandas.DatetimeIndex",
    "IntervalIndex": "~pandas.IntervalIndex",
    "Series": "~pandas.Series",
    "DataFrame": "~pandas.DataFrame",
    "Categorical": "~pandas.Categorical",
    "Path": "~~pathlib.Path",
    # objects with abbreviated namespace (from pandas)
    "pd.Index": "~pandas.Index",
    "pd.NaT": "~pandas.NaT",
}

# numpydoc_show_class_members = False
# Report warnings for all validation checks except the ones listed after "all"
numpydoc_validation_checks = {"all", "ES01", "EX01", "SA01", "SA04"}
# don't report on objects that match any of these regex
numpydoc_validation_exclude = {
    r"\.__repr__$",
    # "cf_xarray.accessor.",
    # "cf_xarray.set_options.",
}

# -- Options for HTML output

html_theme = 'pydata_sphinx_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

