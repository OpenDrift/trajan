"""
Read a position log from a CSV file and convert it to a CF-compatible dataset
=============================================================================
"""
import pandas as pd
import trajan as ta
import matplotlib.pyplot as plt

#%%
# Use the `read_csv` function:
ds = ta.read_csv('bug05_pos.csv.xz', lon='Longitude', lat='Latitude', time='Time', name='Device')
print(ds)

ds.traj.plot(color=None, label=ds.trajectory.values)
plt.legend()
plt.show()

