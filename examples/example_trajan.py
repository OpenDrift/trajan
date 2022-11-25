"""
Demonstrating basic plotting
============================
"""
import cartopy.crs as ccrs
import lzma
import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta
import coloredlogs

###########################################################
# Demonstrating how a trajectory dataset (from OpenDrift)
# can be analysed and plotted with Trajan
###########################################################
coloredlogs.install(level='debug')

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
plt.title('Basic trajectory plot')
plt.show()

##################################################################################
# Demonstrating how the Xarray Dataset can be modified, allowing for
# more flexibility than can be provided through the plotting method of OpenDrift
##################################################################################

# Extracting only the first 10 elements, and every 4th output time steps:
d.isel(trajectory=range(0, 10), time=range(0, len(d.time), 4)).traj.plot()
# TODO: the above title is not shown, since a subset of d is plotted (and not d itself)
plt.title('First 10 elements, and every 4th time steps')
plt.show()

# Plotting a "mean" trajectory on top
###################################################################
# TODO: the below does not work: only the mean trajectory is shown
###################################################################
d.traj.plot(color='red', alpha=0.01)  # Plotting trajectories in red
dmean = d.mean('trajectory', skipna=True)
dmean.traj.plot.lines(color='black', linewidth=5)  # Plotting mean trajectory in black
plt.show()

# Calling set_up_map explicitly
plt.figure()
ax = d.traj.plot.set_up_map(margin=0)
d.traj.plot(ax=ax, color='red', alpha=0.01)  # Plotting trajectories in red
dmean.traj.plot(ax=ax, color='black', alpha=1, linewidth=5)  # Plotting mean trajectory in black
# Plotting the mean trajectory for a sub period in yellow
dmean17nov = d.sel(time=slice('2015-11-17', '2015-11-17 12')).mean('trajectory', skipna=True)
dmean17nov.traj.plot(ax=ax, color='yellow', alpha=1, linewidth=5)
#plt.show()  # TODO: this shows nothing
#ax.get_figure().show()  # TODO: this shows nothing
ax.get_figure().savefig('testplot.png')  # TODO: this produces a figure, but with title from previous
