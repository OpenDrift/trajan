"""
Analysing trajectories in nc_particles format
=============================================
"""

import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta

#%%
# Demonstrating analysis of a trajecory_dataset from GNOME in the nc_particles format
# https://noaa-orr-erd.github.io/nc_particles/nc_particle_standard.html
# This is a contiguous ragged format with a deviation from the CF-convention:
# all elements for a given time are stored together, instead of the time series of individual elements

ds = xr.open_dataset(ta.DATA_DIR + 'gnome_nc_particles.nc')
print(ds)

print(ds.traj)

#%%
# Basic plot
ds.traj.plot(land='mask')

#%%
# Convert from Ragged to Orthogonal
ds_ortho = ds.traj.gridtime('1h')

#%%
# Plotting a mean trajectory
ds_ortho.mean('trajectory').traj.plot(color='r', label='Mean trajectory')

#%%
# Plotting initial and final element locations, and a convex hull
ds_ortho.isel(time=0).traj.plot.scatter(color='k', s=80, zorder=100, label='Initial positions')
ds_ortho.isel(time=-1).traj.plot.scatter(color='b', s=80, zorder=100, label='Final positions')
ds_ortho.isel(time=-1).traj.plot.convex_hull(color='g', label='Convex hull of final positions')
#ds_ortho.isel(time=10).traj.plot.convex_hull(color='y', label='Convex hull after 10 hours')

plt.legend()
plt.title('Sample trajectory dataset from nc_particles repository')
plt.show()

#%%
# Plotting the element concentration after 8 hours
grid = ds_ortho.isel(time=8).traj.make_grid(dx=1000)
ds_concentration = ds_ortho.isel(time=8).traj.concentration(grid)
ds_concentration.number.plot()
plt.title('Element concentration after 10 hours')
plt.show()

#%%
# Basic animation
ds_ortho.traj.animate().set_title('nc_particles sample dataset').show()
