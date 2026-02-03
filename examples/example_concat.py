"""
Concatenating drifter datasets
==============================
"""
import numpy as np
import lzma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
import trajan as ta

#%%
# Importing a dataset with two drifters in the Barents Sea
with lzma.open('barents.nc.xz') as barents:
    ds = xr.open_dataset(barents)
    ds.load()

assert 'obs' in ds.dims

#%%
# Split into two observational datasets for this example

ds = ds.rename(drifter_names='trajectory').traj.condense_obs()
print(ds)

d1 = ds.isel(trajectory=0).traj.to_1d().traj.to_2d()
d2 = ds.isel(trajectory=1).traj.to_1d().traj.to_2d()
print("d1=", d1)
print("d2=", d2)

#%%
# Concatenate two 2D datasets (with observation dimension).

dc = xr.concat((d1, d2), dim='trajectory', join='outer')
dc = dc.traj.condense_obs()
print(dc)

assert np.all(ds.lat.values[~np.isnan(ds.lat.values)] ==
              dc.lat.values[~np.isnan(dc.lat.values)])

#%%
# Concatenating two 1D datasets, with observations at different times.

d1 = d1.traj.to_1d(
)  # trivial conversion since `d1` only contains a single trajectory. No need for gridtime.
d2 = d2.traj.to_1d()  # Also trivial for d2.
print(d1)

assert 'obs' not in d1.dims

#%%
# Concatenating two 1D datasets will cause a lot of NaNs to be inserted.
d1 = d1.drop_duplicates('time')
d2 = d2.drop_duplicates('time')
dc = xr.concat((d1, d2), dim='trajectory', join='outer')
print(dc)

assert np.all(ds.lat.values[~np.isnan(ds.lat.values)] ==
              dc.lat.values[~np.isnan(dc.lat.values)])

#%%
# Converting to 2D and condensing the dataset will give a cleaner result.
dc = xr.concat((d1.traj.to_2d(), d2.traj.to_2d()),
               dim='trajectory', join='outer').traj.condense_obs()
print(dc)
assert dc.sizes['obs'] == ds.sizes['obs']
