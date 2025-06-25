"""
Analysing output from Parcels (Zarr format)
===========================================
"""

import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta

ds = xr.open_dataset('../tests/test_data/parcels.zarr', engine='zarr')
#%%
# Print Xarray dataset
print(ds)

#%%
# Print trajectory specific information about dataset
print(ds.traj)

#%%
# Basic plot
ds.traj.plot(land='mask', margin=1)

# TODO: we must allow no time dimension for the below to work
#ds.mean('trajectory', skipna=True).traj.plot(color='r', label='Mean trajectory')
# In the meantime, we regrid to a regular 1D dataset to allow plotting a mean trajectory
ds = ds.traj.gridtime('4h')
ds.mean('trajectory').traj.plot(color='r', label='Mean trajectory')
plt.legend()
plt.show()

#%%
# Calculating and plotting the concentration of elements, after 6 and 24 hours
grid = ds.traj.make_grid(dx=3000)
ds_conc = ds.traj.concentration(grid)
plt.subplot(1,2,1)
ds_conc.number.isel(time=6).plot(vmin=0, vmax=20)
plt.scatter(ds.isel(time=6).lon, ds.isel(time=6).lat, s=1, color='black')
plt.subplot(1,2,2)
ds_conc.number.isel(time=24).plot(vmin=0, vmax=20)
plt.scatter(ds.isel(time=24).lon, ds.isel(time=24).lat, s=1, color='black')
plt.show()

#%%
# Calculating skillscore
# Defining the first trajectory to be the "true"
ds_true = ds.isel(trajectory=0)
skillscore = ds.traj.skill(expected=ds_true, method='liu-weissberg', tolerance_threshold=1)

#%%
# Plotting trajectories, colored by their skillscore, compared to the "true" trajectory (black)
mappable = ds.traj.plot(land='mask', color=skillscore)
ds_true.traj.plot(color='k', linewidth=3, label='"True" trajectory')
plt.colorbar(mappable=mappable, orientation='horizontal', label='Skillscore per trajectory')
plt.legend()
plt.show()
