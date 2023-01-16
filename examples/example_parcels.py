"""
Analysing output from Parcels (Zarr format)
===========================================
"""

import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta

ds = xr.open_dataset('../tests/test_data/parcels.zarr', engine='zarr')
print(ds)
ds.traj.plot(land='mask', margin=2)
#ds.mean('trajectory', skipna=True).traj.plot(color='r', label='Mean trajectory')

plt.show()
