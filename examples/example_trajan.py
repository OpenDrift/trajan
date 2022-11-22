import os
import lzma
import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta

###########################################################
# Demonstrating how a trajectory dataset (from OpenDrift)
# can be analysed and plotted with Trajan
###########################################################

###################################################################################################
# Importing a trajectory dataset from a simulation with OpenDrift.
# decode_coords is needed so that lon and lat are not interpreted as coordinate variables.
with lzma.open('openoil.nc.xz') as oil:
    d = xr.open_dataset(oil, decode_coords=False)
    d.load()
    # Requirement that status>=0 is needed since non-valid points are not masked in OpenDrift output
    d = d.where(d.status>=0)  # only active particles
###################################################################################################

# Displaying a basic plot of trajectories
d.traj.plot()
# which is equivalent to
ta.plot(d)

# Creating a plot, but adding customization (title) before saving to file
ax, fig, gcrs = ta.plot(d, show=False)
ax.set_title('Adding custom title')
fig.savefig('testplot.png')

##################################################################################
# Demonstrating how the Xarray Dataset can be modified, allowing for
# more flexibility than can be provided through the plotting method of OpenDrift
##################################################################################

# Extracting only the first 100 elements, and every 4th output time steps:
dsub = d.isel(trajectory=range(0, 100), time=range(0, len(d.time), 4))
dsub.traj.plot()

# Plotting a "mean" trajectory
dmean = d.mean('trajectory', skipna=True)
dmean.traj.plot(trajectory_kwargs={'color': 'red', 'linewidth': 5})

# Using set_up_map only, and plotting trajectories manually
ax, fig, gcrs = ta.set_up_map(d, land_color='green')
ax.plot(d.lon.T, d.lat.T, color='red', alpha=0.01, transform=gcrs)  # Plotting trajectories in red
ax.plot(dmean.lon.T, dmean.lat.T, color='black', alpha=1, linewidth=5, transform=gcrs)  # Plotting mean trajectory in black
# Plotting the mean trajectory for a sub period in yellow
dmean17nov = d.sel(time=slice('2015-11-17', '2015-11-17 12')).mean('trajectory', skipna=True)
ax.plot(dmean17nov.lon.T, dmean17nov.lat.T, color='yellow', alpha=1, linewidth=5, transform=gcrs)
plt.show()
