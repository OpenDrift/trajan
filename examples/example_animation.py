"""
Animating drifter trajectories
===============================

Trajan supports animating trajectory datasets using a builder pattern.
Start with ``ds.traj.animate()``, chain configuration methods, then call
``.show()`` to display interactively or ``.save()`` to write a file.
"""
import logging
import lzma
import numpy as np
import xarray as xr
import coloredlogs
import trajan as ta

coloredlogs.install(level=logging.DEBUG, logger=logging.getLogger('trajan'))

#%%
# Load the Barents Sea drifter dataset, drop short gaps and grid to 6-hourly.
with lzma.open('barents.nc.xz') as barents:
    ds = xr.open_dataset(barents)
    ds.load()

ds = ds.traj.drop_where(ds.traj.time_to_next() < np.timedelta64(5, 'm'))
ds = ds.traj.gridtime('6h')
speed = ds.traj.speed()

#%%
# Basic animation — just call ``.animate()`` and ``.show()``.
# The title shows the current timestamp automatically.
ds.traj.animate().set_title('Barents Sea drifters').show()

#%%
# Colour particles by drift speed.
# Pass the DataArray directly to ``color_by`` — no need to add it to the dataset.
(ds.traj.animate()
    .color_by(speed, cmap='plasma', vmin=0, vmax=1,
              label='Speed  [m/s]')
    .set_title('Drifter speed')
    .show())

#%%
# Show static trajectory lines in the background so the full path
# of each drifter is always visible.
(ds.traj.animate()
    .color_by(speed, cmap='plasma', vmin=0, vmax=1,
              label='Speed  [m/s]')
    .show_trajectories(alpha=0.15)
    .set_title('Speed with trajectory history')
    .show())

#%%
# Save the animation to an MP4 file (faster and smaller than GIF).
(ds.traj.animate()
    .color_by(speed, cmap='plasma', vmin=0, vmax=1,
              label='Speed  [m/s]')
    .show_trajectories(alpha=0.2)
    .set_title('Barents Sea drifters')
    .set_fps(10)
    .save('barents_drifters.mp4'))

#%%
# .. image:: /gallery/animations/example_animation_0.gif
