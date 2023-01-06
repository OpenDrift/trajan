"""
Analysing a drifter dataset
============================
"""
import numpy as np
import lzma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import trajan as ta

#%%
# Importing a dataset with two drifters in the Barents Sea
with lzma.open('barents.nc.xz') as barents:
    ds = xr.open_dataset(barents)
    ds.load()

#%%
# This follows the CF convention for trajectories
# https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_multidimensional_array_representation_of_trajectories
print(ds)

#%%
# A basic plot can be made very simply with Trajan
ds.traj.plot()
plt.show()

#%%
# The figure can be customized, combining functionality from Trajan, Xarray and Matplotlib
ds.isel(trajectory=1).traj.plot(color='b', label=ds.drifter_names[1].values,
                                land='mask', margin=3)
ds.isel(trajectory=0).traj.plot(color='r', label=ds.drifter_names[0].values)
plt.legend()
plt.title('Two drifters in the Barents Sea')
plt.show()

#%%
# Trajan includes methods to e.g. calculate the speed of the drifters
speed = ds.traj.speed()
print(f'Max speed {speed.max().values} m/s')

#%%
# We see that the max speed > 1000 m/s, which is a numerical error
# due to some cases with GPS positions reported with very small time interval.
# By removing all positions where time interval < 5 min, we avoid this problem.
ds = ds.traj.drop_where(ds.traj.time_to_next() < np.timedelta64(5, 'm'))
speed = ds.traj.speed()
print(f'Max speed {speed.max().values} m/s')

#%%
# Likewise, one can insert breaks (NaN) in the trajectories
# whenever the time between points exceed a desired threshold, e.g. 3 hours
ds = ds.traj.insert_nan_where(ds.traj.time_to_next()>np.timedelta64(3, 'h'))

#%%
# Plotting trajectories colored by drifter speed
mappable = ds.traj.plot(color=speed)

cb = plt.gcf().colorbar(mappable,
          orientation='horizontal', pad=.05, aspect=30, shrink=.8, drawedges=False)
cb.set_label('Speed  [m/s]')
plt.title('Trajectories colored by drift speed')
plt.show()

#%%
# Histogram of drifter speeds.
speed = ds.traj.speed()
plt.hist(speed.values[~np.isnan(speed.values)], 100)
plt.xlabel('Drifter speed  [m/s]')
plt.ylabel('Number')
plt.show()

#%%
# The peak at speed=0m/s is from the period where one of the drifters are on land (Hopen island).
# This can be removed simply:
speed = speed.where(speed>0.01)
plt.hist(speed.values[~np.isnan(speed.values)], 100)
plt.xlabel('Drifter speed  [m/s]')
plt.ylabel('Number')
plt.show()

#%%
# The positions of GPS based drifters are normally given at slightly irregular time intervals,
# as drifters may be without GPS coverage for periods, and may get GPX fixes irregularly.
# TrajAn contains the method `gridtime` to interpolate positions to a regular time intervel, e.g. hourly:
dh = ds.traj.gridtime('1h')
ds.traj.plot(color='r', label='raw data', land='mask')
dh.traj.plot(color='b', label='hourly')
plt.gca().set_extent([23.8, 25.0, 76.8, 77], crs=ccrs.PlateCarree())  # Zooming in to highliht differences
plt.legend()
plt.show()

#%%
# Having the dataset on a regular time interval makes it simpler to compare with e.g. drift models,
# and also makes some analyses simpler
dh.isel(trajectory=0).traj.plot(color='r', label='Full trajectory')
dh.isel(trajectory=0).sel(time=slice('2022-10-10', '2022-10-12')).traj.plot(
            color='k', linewidth=2, label='10-12 Oct 2022')
plt.legend()
plt.show()

#%%
# The original dataset had two dimensions `(trajectory, obs)` (see top of page), and time is a 2D variable.
# For the gridded dataset (as with datasets imported from some trajectory models), dimensions are `(trajectory, time)` and time is a 1D Xarray dimension coordinate
print(dh)

#%%
# Plotting the velocity spectrum for one drifter
dh.isel(trajectory=1).traj.velocity_spectrum().plot()
plt.xlim([0, 30])
plt.show()
