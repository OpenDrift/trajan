"""
Analysing output from Parcels (Zarr format)
===========================================
"""

import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta

ds = xr.open_dataset('../tests/test_data/parcels.zarr', engine='zarr')
#%%
# Print Xarray dataset
print(ds)

#%%
# Print trajectory specific information about dataset
print(ds.traj)

#%%
# Basic plot
ds.traj.plot(land='mask', margin=1)
# TODO: we must allow no time dimension for the below to work
#ds.mean('trajectory', skipna=True).traj.plot(color='r', label='Mean trajectory')

plt.show()
