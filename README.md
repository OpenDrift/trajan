# Trajectory analysis ( TRAJAN )

Trajan is a Python package with functionality to handle trajectory datasets following the CF-conventions on trajectories:
http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#trajectory-data
Trajectory datasets contain position time series from e.g. drifting buoys, or output from lagrangian models.

In addition to various stand-alone methods, trajan adds an extension of the Xarray Dataset, which is added automatically after

`>>> import traj`

This contains various analysis and visualization methods available within the namespace `traj`:

`>>> ds.traj.<method>`

This allows combining the general functionality of Xarray (e.g. subsetting) with functionality specific to trajectory datasets, e.g.:

`>>> ds.isel(trajectory=range(0,100)).sel(time=slice('2022-11-10', '2022-11-17')).mean('trajectory', skipna=True).traj.plot()`

which will produce a plot of the mean of the first 100 trajectories in a dataset, over a chosen period of 7 days. Here the specific plotting method of the Trajan was used, which also adds a landmask as background, but if `.traj` was omitted from the end of the command, the general Xarray plotting method would be used.

The above example assumes that obs (time) is constant for all trajectories in the dataset, which is the case for e.g. output from lagrangian models.
For a collection of “raw data” of a set of ocean drifters, time will generally be a 2D variable (obs, trajectory), as each drifter may sample its position at different times. Such datasets may be interpolated to a common, regular time, with e.g.:

`>>> ds_obs_regular = ds_obs.traj.gridtime(‘1H’)`

to interpolate to a fixed hourly times step, or

`>>> ds_obs_regular = ds_obs.traj.gridtime(ds_model)`

to interpolate and subselect a drifter dataset to the same time step and period as the output of a trajectory model (ds_model). This facilitates e.g. comparison of modelled and observed trajectories.
