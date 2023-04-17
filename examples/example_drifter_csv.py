"""
Read a position log from a CSV file and convert it to a CF-compatible dataset
=============================================================================
"""
import pandas as pd
import trajan as ta
import matplotlib.pyplot as plt

#%%
# Alternative 1: Use the built in `read_csv` method:
ds = ta.read_csv('bug05_pos.csv.xz', lon='Longitude', lat='Latitude', time='Time', name='Device')
print(ds)

#%%
# Alternative 2: Manually construct the dataset:

#%%
# Read a CSV file with positions using Pandas.
ds = pd.read_csv('bug05_pos.csv.xz', parse_dates=['Time'])

#%%
# Make a synthetic drifter placed slightly east of the original drifter, and concatenate it to the
# original dataset.
ds2 = ds.copy()
ds2['Device'] = 'dev_synthetic'
ds2['Longitude'] = ds2['Longitude'] + 0.001
ds = pd.concat((ds, ds2))

#%%
# Rename the index to `obs`, and rename the other columns to the standard coordinate names.
ds.index.names = ['obs']
ds = ds.rename(columns={'Latitude': 'lat', 'Longitude': 'lon', 'Time': 'time', 'Device': 'trajectory'})
ds.time = ds['time'].dt.tz_convert(None)

#%%
# Classify trajectories based on drifter_names.
ds = ds.set_index(['trajectory', ds.index])

#%%
# Convert the Pandas DataFrame to Xarray Dataset
ds = ds.to_xarray()

#%%
# Simplify the drifter_names variable: It is only dependent on the trajectory dimension.
#
# We select the name from the first observation from each trajectory (and drop the 'obs' dimension).
ds['trajectory'] = ds.drifter_names.isel(obs=0)

#%%
# Print the dataset and plot it. See the `example_drifters` example for how to continue analyzing a drifter dataset.
#
# We plot with `color=None` to use normal matplotlib color cycling and not a single color for all drifters. This may be changed to be the default in the furture.
print(ds)

ds.traj.plot(color=None, label=ds.trajectory.values)
plt.legend()
plt.show()

