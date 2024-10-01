Quickstart: read and plot data
------------------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import xarray as xr
   import trajan as _

   ds = xr.open_dataset('drifter_dataset.nc')

   ds.traj.plot()
   plt.show()
