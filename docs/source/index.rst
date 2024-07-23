Trajectory Analysis (TrajAn)
===================================

TrajAn is a Python package with functionality to handle trajectory datasets following the `CF-conventions on trajectories <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#trajectory-data>`_.

Trajectory datasets contain position time series from e.g. drifting buoys, or output from lagrangian models.

The source code is available on https://github.com/OpenDrift/trajan

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

_TrajAn_ is an `Xarry extension <https://docs.xarray.dev/en/stable/>`_. On drifter (or trajectory) datasets you can use the `.traj` accessor on `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_ s. In order to register the accessor _trajan_ needs to be imported:

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

Generally, TrajAn supports two types of data layout: 1) trajectories sampled at different times (unstructured or irregular grid, almost every dataset from real observations. And 2) trajectories sampled at uniform (or a regular grid, typical the output from a model). We often refer to the first type as _2D_ since time is a function of trajectory and observation, while the second type is _1D_ and time is only a function of observation. __TrajAn__ will detect which type of dataset you have and you will have access to the appropriate methods for the type data layout.

While the first type (_2D_) is more general it often limits analysis that require trajectories to be sampled at the same points, you can therefor convert a _2D_ dataset to _1D_ by using :meth:`trajan.traj2d.Traj2d.gridtime`.

Methods applicable to both types of datasets can be found in: :mod:`trajan.traj`, methods for _1D_ datasets: :mod:`trajan.traj.traj1d`, and _2D_: :mod:`trajan.traj.traj2d`. All methods are forwarded to the accessor, so you call the methods on `Dataset.traj`:

.. code-block:: python

   ds = ds.traj.gridtime('1H')   # grid dataset to every hour (a 2D method)
   ds.traj.plot()                # plot dataset


Generic plotting is available in the standard `Xarray` way, and strives to stay as close to `matplotlib` as possible: (:meth:`trajan.trajectory_accessor.TrajAccessor.plot`) and animation (:meth:`trajan.trajectory_accessor.TrajAccessor.animate`).

Contents
--------

.. toctree::
   :maxdepth: 2

   index
   gallery/index
   autoapi/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :doc:`autoapi/index`
