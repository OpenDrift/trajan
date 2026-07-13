"""
Read a position log from a CSV file and convert it to a CF-compatible dataset
=============================================================================
"""
import trajan as ta
import matplotlib.pyplot as plt

#%%
# Use the `read_csv` function:
ds = ta.read_csv(ta.DATA_DIR + 'omb/bug05_pos.csv', lon='Longitude', lat='Latitude', time='Time', name='Device')

# Filter to remove segments where drifter was transported
ds = ds.traj.filter(method='speed', max_speed=3)
print(ds)
print(ds.traj)

ds.traj.plot(color=None, label=ds.trajectory.values)
plt.legend()
plt.show()
