"""
Calculating gridded concentrations
==================================
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import trajan as _

#%%
# Demonstrating calculation and plotting of gridded concentrations

#%%
# Importing a trajectory dataset from an oil drift simulation (OpenOil)
ds = xr.open_dataset('opv2025_subset.nc')

#%%
# Make a grid covering this dataset with a horizontal resolution of 200m
grid = ds.traj.make_grid(dx=200)
print(grid)

#%% 
# Calculate the concentration number of elements for this grid
ds_c = ds.traj.concentration(grid)
print(ds_c)

#%% 
# Plot the number of particles and concentration (number/area) per grid cell at a given time.
ds_c.isel(time=12).number.plot()
plt.show()

#%% 
# Plot the number concentration (number/area) per grid cell at a given time.
# Since pixels have nearly the same size, this is nearly proportional to the previous plot.
ds_c.isel(time=12).number_area_concentration.plot()
plt.show()

#%% 
# We see that the cell area decreases slightly sway from equator.
# This is accounted for when calculating area/volume concentration.
ds_c.cell_area.plot()
plt.show()


#%% 
# Making a new grid including a vertical dimension (3D), with highest resolution near the surface (z=0)
z=[-0.01, -1, -3, -5, -7.5, -10, -20, -30]
grid3d = ds.traj.make_grid(dx=400, z=z)

#%% 
# Calculate the concentration of elements for this new grid, also weighted with element property "mass_oil"
weights = 'mass_oil'
ds = ds.isel(time=12)  # Pre-selecting a single time before calculating concentrations
ds_c = ds.traj.concentration(grid3d, weights=weights)

#%% 
# Plot the oil concentration (mass/volume, kg/m3) at depths 1-3m and 20-30m
plt.subplot(1,2,1)
cbar_kwargs={'orientation': 'horizontal', 'shrink': .9}
ds_c.mass_oil_volume_concentration.sel(z=-2).plot(vmin=0, vmax=4e-5, cbar_kwargs=cbar_kwargs)
plt.gca().set_aspect('equal', 'box')
plt.subplot(1,2,2)
ds_c.mass_oil_volume_concentration.sel(z=-25).plot(vmin=0, vmax=4e-5, cbar_kwargs=cbar_kwargs)
plt.gca().set_aspect('equal', 'box')
plt.gca().yaxis.set_label_position('right')
plt.gca().yaxis.tick_right()
plt.show()

#%% 
# Plot the vertical profiles of oil concentration versus depth
oil_mass_profile = ds_c.mass_oil_volume_concentration.median(dim='lat')
print(oil_mass_profile)
oil_mass_profile.plot.line(
    y='z', add_legend=False, xlim=(0, None), ylim=(None, 0))
plt.xlabel('Oil concentration [kg / m3]')
plt.show()
