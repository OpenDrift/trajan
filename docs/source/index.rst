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

**Install from source (recommended if you want to modify TrajAn code)**

.. code-block:: console

   $ git clone git@github.com:OpenDrift/trajan.git
   $ cd trajan
   $ pip install -e .

Remember to re-install each time you have done an edit during the development process.

Usage
-----

*TrajAn* is an `Xarry extension <https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_. On drifter (or trajectory) datasets you can use the `.traj` accessor on `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_. In order to register the accessor, `trajan` needs to be imported:

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

For Ragged datasets time is a 2D array with dimensions for trajectory and observation/time, while for Orthognal datasets time is a 1D array common for all trajectories.

TrajAn will detect which type of dataset you have and you will have access to the appropriate methods for the type data layout. `Contiguous ragged <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_contiguous_ragged_array_representation_of_trajectories>`_ and `nc_particles <https://noaa-orr-erd.github.io/nc_particles/nc_particle_standard.html>`_ format are internally converted to Ragged format.

While the Ragged format is more general it often limits analysis that require trajectories to be sampled at the same points, you can therefor convert a Ragged dataset to Orthogonal by using :meth:`xarray.Dataset.traj.gridtime`.

Methods applicable to both types of datasets can be found in: :class:`trajan.accessor.TrajA`, methods for Orthogonal datasets: :class:`trajan.traj.orthogonal.Orthogonal`, and Ragged: :class:`trajan.traj.ragged.Ragged`. All methods are forwarded to the accessor, so you call the methods on :mod:`ds.traj`:

.. code-block:: python

   ds = ds.traj.gridtime('1h')   # grid dataset to every hour
   ds.traj.plot()                # plot dataset


Generic plotting is available in the standard `Xarray` way, and strives to stay as close to `matplotlib` as possible:
:meth:`ds.traj.plot <xarray.Dataset.traj.plot>`
TrajAn also contains an animation builder mechanism the can be chained:
:meth:`ds.traj.animate <xarray.Dataset.traj.animate`

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
