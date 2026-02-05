"""
Read variables from a model along the trajectories of drifters.
=============================================================================
"""

import xarray as xr
import numpy as np
import cf_xarray as _
import pyproj
import trajan as ta
import matplotlib.pyplot as plt

print(ta.versions())

#%%
# Open drifter dataset from a CSV file
ds = ta.read_csv('bug05_pos.csv.xz',
                 lon='Longitude',
                 lat='Latitude',
                 time='Time',
                 name='Device')
print(ds)

#%%
# Open the Norkyst800 model
nk = xr.open_dataset(
    'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be')

#%%
# Use cf-xarray to get the CRS
gm = nk.cf['grid_mapping']
nk_crs = pyproj.CRS.from_cf(gm.attrs)
print(nk_crs)

#%%
# Grid the drifter dataset to the timesteps of the model.
times = nk.sel(time=slice('2022-05-10', '2022-05-20')).time.values
ds = ds.traj.gridtime(times)
ds = ds.dropna('time', how='all')

print(ds)

#%%
# Transform the drifter dataset to the CRS of the model
dst = ds.traj.transform(nk_crs)
print(dst.x)

#%%
# Extract the values of a variable for the trajectory
temp = nk.isel(depth=0).sel(time=dst.time,
                            X=dst.x,
                            Y=dst.y,
                            method='nearest').temperature
print(temp)

#%%
# Notice that the `lat` and `lon` variables from Norkyst match the original `lat` and `lon` from the dataset.

plt.figure()
temp.plot()
plt.show()
