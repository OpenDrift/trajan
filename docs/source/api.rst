API
===

.. currentmodule:: trajan

Top-level API
-------------

.. autosummary::
    :toctree: generated/
    :caption: Loading datasets

    read_csv
    from_dataframe
    trajectory_dict_to_dataset (deprecated)

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
   :caption: Datasets

    Dataset.traj.plot
    Dataset.traj.animate
    Dataset.traj.tx
    Dataset.traj.ty
    Dataset.traj.tlon
    Dataset.traj.tlat
    Dataset.traj.crs
    Dataset.traj.ccrs

.. _dsmeth:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.traj.transform
    Dataset.traj.transformer
    Dataset.traj.set_crs
    Dataset.traj.assign_cf_attrs
    Dataset.traj.index_of_last
    Dataset.traj.speed
    Dataset.traj.time_to_next
    Dataset.traj.distance_to
    Dataset.traj.distance_to_next
    Dataset.traj.azimuth_to_next
    Dataset.traj.length
    Dataset.traj.skill
    Dataset.traj.velocity_components
    Dataset.traj.velocity_spectrum
    Dataset.traj.convex_hull
    Dataset.traj.convex_hull_contains_point
    Dataset.traj.get_area_convex_hull
    Dataset.traj.gridtime
    Dataset.traj.sel
    Dataset.traj.seltime
    Dataset.traj.iseltime
    Dataset.traj.append
    Dataset.traj.crop
    Dataset.traj.contained_in
    Dataset.traj.is_1d
    Dataset.traj.is_2d
    Dataset.traj.to_1d
    Dataset.traj.to_2d
    Dataset.traj.condense_obs

.. currentmodule:: trajan

Plotting
--------

.. autosummary::
   :toctree: generated/
   :caption: Plotting

   plot.Plot
   animation.Animation




