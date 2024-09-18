API
===

.. currentmodule:: trajan

Top-level API
-------------

.. autosummary::
    :toctree: generated/

    read_csv
    from_dataframe
    trajectory_dict_to_dataset (deprecated)

.. Geometries
.. ----------
.. .. autosummary::
..     :toctree: generated/

..     geometry.decode_geometries
..     geometry.encode_geometries
..     geometry.shapely_to_cf
..     geometry.cf_to_shapely
..     geometry.GeometryNames

.. currentmodule:: xarray

.. DataArray
.. ---------

.. .. _daattr:

.. Attributes
.. ~~~~~~~~~~

.. .. autosummary::
..    :toctree: generated/
..    :template: autosummary/accessor_attribute.rst

..     DataArray.cf.axes
..     DataArray.cf.cell_measures
..     DataArray.cf.cf_roles
..     DataArray.cf.coordinates
..     DataArray.cf.formula_terms
..     DataArray.cf.grid_mapping_name
..     DataArray.cf.is_flag_variable
..     DataArray.cf.standard_names
..     DataArray.cf.plot


.. .. _dameth:

.. Methods
.. ~~~~~~~

.. .. autosummary::
..    :toctree: generated/
..    :template: autosummary/accessor_method.rst

..     DataArray.cf.__getitem__
..     DataArray.cf.__repr__
..     DataArray.cf.add_canonical_attributes
..     DataArray.cf.differentiate
..     DataArray.cf.guess_coord_axis
..     DataArray.cf.keys
..     DataArray.cf.rename_like

.. Flag Variables
.. ++++++++++++++

.. cf_xarray supports rich comparisons for `CF flag variables`_. Flag masks are not yet supported.

.. .. autosummary::
..    :toctree: generated/
..    :template: autosummary/accessor_method.rst

..     DataArray.cf.__lt__
..     DataArray.cf.__le__
..     DataArray.cf.__eq__
..     DataArray.cf.__ne__
..     DataArray.cf.__ge__
..     DataArray.cf.__gt__
..     DataArray.cf.isin


Dataset
-------

.. _dsattr:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.traj.tx
    Dataset.traj.ty
    Dataset.traj.tlon
    Dataset.traj.tlat

.. _dsmeth:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.traj.transform
    Dataset.traj.itransform


