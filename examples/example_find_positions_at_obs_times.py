"""
Finding the closest or interpolated positions at given times (e.g. wave observations).
=======================================================================================
"""

from pathlib import Path
from trajan.readers.omb import read_omb_csv
import xarray as xr
import coloredlogs

coloredlogs.install(level='debug')

#%%
# Read the data
data = Path.cwd().parent / "tests" / "test_data" / "csv" / "omb3.csv"
ds = read_omb_csv(data)
print(ds)

#%%
# The wave data in the variable `pHs0` is given along a different observation dimension.
# Because it is also a observation, i.e. in the style of 2D trajectory
# datasets, we need to iterate over the trajectories:


def gridwaves(tds):
    t = tds[['lat', 'lon',
             'time']].traj.gridtime(tds['time_waves_imu'].squeeze())
    return t.traj.to_2d(obs_dim='obs_waves_imu')


dsw = ds.traj.trajectories().map(gridwaves)

print(dsw)

#%%
# We now have the positions interpolated to the IMU (wave) observations. We
# could also merge these together to one dataset again:

ds = xr.merge((ds, dsw.rename({
    'lon': 'lon_waves',
    'lat': 'lat_waves'
}).drop('time')))
print(ds)
