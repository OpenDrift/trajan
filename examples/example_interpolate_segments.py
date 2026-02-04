"""
Interpolating trajectories with holes (NaN)
===========================================
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta

#%%
# Emulating a drifter dataset with gaps and irregularities
time_interval_seconds = [3600, 3600, 3600, 3600*3, 3600, 3600, 3600, 3600]
time_interval_seconds = np.array(time_interval_seconds) + np.random.rand(len(time_interval_seconds))*200
time = np.datetime64('2026-02-01T00:00:00') + time_interval_seconds.cumsum().astype('timedelta64[s]') 
lon = [3, 3.1, 3.2, 3.4, 3.5, 3.3, 3.4, 3.4]
lat = [60, 60, 60.04, 60.05, 60.1, 60.08, 60.12, 60.1]
lon = np.array(lon) + np.random.rand(len(lon))*.01
lat = np.array(lat) + np.random.rand(len(lat))*.01
lon = np.vstack([lon, lon])
lat = np.vstack([lat, lat+.1])
# Add second break in trajectory 2
time2 = time+np.timedelta64(33, 's')
time2[5:] += np.timedelta64(3600*2, 's')
time = np.vstack([time, time2])

ds = xr.Dataset(
    {
        "lon": (("trajectory", "obs"), lon),  # Data variable
        "lat": (("trajectory", "obs"), lat),  # Data variable
        "time": (("trajectory", "obs"), time),  # Time coordinate
    },
    coords = {
        'obs': np.arange(lon.shape[1]),
        'trajectory': [1, 2],
        }
)

#%%
# Plotting the original trajectories along with interpolated to 10 minutes regular interval.
# Interpolation also fills the larger gaps with 3 hour interval, which is questionable
ds.traj.plot(color='k', linewidth=1, marker='o', markersize=10, land=None, label='Original')
dh = ds.traj.gridtime('600s')
dh.traj.plot(color='r', linewidth=1, land=None, marker='.', label='Interpolated to regular 10 minutes')
plt.legend()
plt.show()

#%%
# We insert breaks (NaN) in trajectories whenever time to next observations > 2 hours
# The subsequent interpolation (gridtime) now avoids the interpolation across the gaps
dsb = ds.traj.insert_nan_where(ds.traj.time_to_next()>np.timedelta64(2, 'h'))
dsb.traj.plot(color='y', linewidth=5, land=None, label='Original trajectories with NaN inserted')
dh = dsb.traj.gridtime('600s')
dh.traj.plot(color='r', linewidth=2, land=None, marker='.', label='Interpolated to regular 10 minutes')
plt.legend()
plt.show()

print(dh)
