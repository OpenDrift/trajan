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
ds = xr.open_dataset('openoil.nc')

#%%
# Displaying some basic information about this dataset
print(ds.traj)

#%%
# Plotting trajectories, colored by oil viscosity
mappable = ds.traj.plot(color=ds.viscosity, alpha=1)
plt.title('Oil trajectories')
plt.colorbar(mappable, orientation='horizontal', label=f'Oil viscosity  [{ds.viscosity.units}]')
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
