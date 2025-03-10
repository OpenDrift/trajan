"""
Analysing output from OpenDrift
===============================
"""
import lzma
import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta

#%%
# Demonstrating how a trajectory dataset (from OpenDrift)
# can be analysed and plotted with Trajan

#%%
# Importing a trajectory dataset from a simulation with OpenDrift.
# decode_coords is needed so that lon and lat are not interpreted as coordinate variables.
with lzma.open('openoil.nc.xz') as oil:
    ds = xr.open_dataset(oil, decode_coords=False)
    ds.load()
    # Requirement that status>=0 is needed since non-valid points are not masked in OpenDrift output
    ds = ds.where(ds.status>=0)  # only active particles

#%%
# Displaying some basic information about this dataset
print(ds.traj)

#%%
# Making a basic plot of trajectories
ds.traj.plot()
plt.title('Basic trajectory plot')
plt.show()

#%%
# Demonstrating how the Xarray Dataset can be modified, allowing for
# more flexibility than can be provided through the plotting method of OpenDrift

#%%
# Extracting only the first 10 elements, and every 4th output time steps:
ds.isel(trajectory=range(0, 10), time=range(0, len(ds.time), 4)).traj.plot()
plt.title('First 10 elements, and every 4th time steps')
plt.show()

#%%
# Plotting a "mean" trajectory on top, with a sub period in yellow
ds.traj.plot(color='red', alpha=0.01, land='fast')  # Plotting trajectories in red, and with landmask as land.
dmean = ds.mean('trajectory', skipna=True)
dmean.traj.plot.lines(color='black', linewidth=5)  # Plotting mean trajectory in black
dmean.sel(time=slice('2015-11-17', '2015-11-17 12')).traj.plot(color='yellow', alpha=1, linewidth=5)
plt.tight_layout()
plt.show()
