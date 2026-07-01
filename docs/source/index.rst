Trajectory Analysis (TrajAn)
===================================

TrajAn is a Python package with functionality to handle trajectory datasets following the `CF-conventions on trajectories <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#trajectory-data>`_.

Trajectory datasets contain position time series from e.g. drifting buoys, or output from lagrangian models.

The source code is available on GitHub: https://github.com/OpenDrift/trajan

Installation
------------

**Install from internet package sources (recommended for users)**

.. code-block:: console

   $ conda install -c conda-forge trajan


or

.. code-block:: console

   $ pip install trajan

**Install from source (recommended to develop for Trajan)**

.. code-block:: console

   $ cd trajan  # move to the location of the trajan root, containing the pyproject.toml file
   $ pip install .

Remember to re-install each time you have done an edit during the development process.

Usage
-----

*TrajAn* is an `Xarry extension <https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_. On drifter (or trajectory) datasets you can use the `.traj` accessor on `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_ s. In order to register the accessor, _trajan_ needs to be imported:

.. code-block:: python

   import matplotlib.pyplot as plt
   import xarray as xr
   import trajan as _

   ds = xr.open_dataset('drifter_dataset.nc')

   ds.traj.plot()
   plt.show()

   speed = ds.traj.speed()
   print(f'Max speed {speed.max().values} m/s')


Trajectory datasets from different models and observations tend to have many small differences. TrajAn expects the dataset to be `CF-compliant <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#trajectory-data>`_. However, the standard does leave some room for interpretation.

Generally, TrajAn supports two types of data layout:
  1) Ragged: trajectories sampled at different times (unstructured or irregular grid), almost every dataset from real observations.
  2) Orthogonal: trajectories sampled at uniform (or regular) grid, typical the output from a model.

For Ragged datasets is time function of trajectory and observation, while for Orthognal datasets time is only a function of observation.

TrajAn will detect which type of dataset you have and you will have access to the appropriate methods for the type data layout.

While the first type (Ragged) is more general it often limits analysis that require trajectories to be sampled at the same points, you can therefor convert a Ragged dataset to Orthogonal by using :meth:`trajan.traj.Ragged.gridtime`.

Methods applicable to both types of datasets can be found in: :mod:`trajan.traj`, methods for Orthogonal datasets: :mod:`trajan.traj.trajOrthogonal`, and Ragged: :mod:`trajan.traj.trajRagged`. All methods are forwarded to the accessor, so you call the methods on `Dataset.traj`:

.. code-block:: python

   ds = ds.traj.gridtime('1H')   # grid dataset to every hour
   ds.traj.plot()                # plot dataset


Generic plotting is available in the standard `Xarray` way, and strives to stay as close to `matplotlib` as possible: (:meth:`trajan.trajectory_accessor.TrajAccessor.plot`) and animation (:meth:`trajan.trajectory_accessor.TrajAccessor.animate`).

Contents
--------

.. toctree::
   :maxdepth: 2

   user_guide/index
   gallery/index

   API Reference <api>


Indices and tables
==================

* :ref:`genindex`



.. |date| date::
.. |time| date:: %H:%M

Last Updated on |date| at |time|
