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

#%% Open drifter dataset from a CSV file
ds = ta.read_csv('bug05_pos.csv.xz',
                 lon='Longitude',
                 lat='Latitude',
                 time='Time',
                 name='Device')
print(ds)

#%% Open the Norkyst800 model
nk = xr.open_dataset(
    'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be')

#%% Use cf-xarray to get the CRS
gm = nk.cf['grid_mapping']
nk_crs = pyproj.CRS.from_cf(gm.attrs)
print(nk_crs)

#%% Grid the drifter dataset to the timesteps of the model.
times = nk.sel(time=slice('2022-05-10', '2022-05-20')).time.values
ds = ds.traj.gridtime(times)
ds = ds.dropna('time', how='all')

#%% Transform the drifter dataset to the CRS of the model
tx, ty = ds.traj.transform(nk_crs, ds.traj.tx, ds.traj.ty)

tx = xr.DataArray(tx, dims=['trajectory', 'time'])
ty = xr.DataArray(ty, dims=['trajectory', 'time'])

#%% Extract the values of a variable for the trajectory
temp = nk.isel(depth=0).sel(time=ds.time,
                            X=tx,
                            Y=ty,
                            method='nearest').temperature
print(temp)

plt.figure()
temp.plot()
plt.show()
