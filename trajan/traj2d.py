import xarray as xr
import numpy as np
import pandas as pd
import logging

from .traj import Traj, ensure_time_dim

logger = logging.getLogger(__name__)


def __require_obs_dim__(f):
    """This decorator is for methods of Traj that require a time or obs dimension to work."""

    def wrapper(*args, **kwargs):
        if args[0].obs_dim is None:
            raise ValueError(f'{f} requires an obs or time dimension')
        return f(*args, **kwargs)

    return wrapper


class Traj2d(Traj):
    """
    A unstructured dataset, where each trajectory may have observations at different times. Typically from a collection of drifters.
    """

    def __init__(self, ds, trajectory_dim, obs_dim, time_varname):
        super().__init__(ds, trajectory_dim, obs_dim, time_varname)

    def timestep(self, average=np.nanmedian):
        """
        Calculate the median time step between observations in seconds.

        Parameters
        ----------
        average : callable, optional
            Function to calculate the average time step, by default `np.nanmedian`.

        Returns
        -------
        xarray.DataArray
            Median time step between observations.
            Attributes:
            - units: seconds
        """
        #td = np.diff(self.ds.time, axis=1) / np.timedelta64(1, 's')
        td = self.ds.time.diff(dim=self.obs_dim)
        td = average(td)
        return xr.DataArray(td, name="timestep", attrs={"units": "seconds"})

    def time_to_next(self):
        """
        Calculate the time difference to the next observation.

        Returns
        -------
        xarray.DataArray
            Time difference to the next observation with the same dimensions as the dataset.
            Attributes:
            - units: seconds
        """
        time = self.ds.time
        lenobs = self.ds.sizes[self.obs_dim]
        td = time.diff(dim=self.obs_dim)
        td = xr.concat((td, td.isel(obs=-1)), dim=self.obs_dim)  # Repeat last time step
        td.coords[self.obs_dim] = time[self.obs_dim]  # Same time index as initial
        return td.astype("timedelta64[s]").rename("time_to_next").assign_attrs({"units": "seconds"})

    def is_1d(self):
        return False

    def is_2d(self):
        return True

    def insert_nan_where(self, condition):
        """
        Insert NaN values in trajectories after given positions, shifting the rest of the trajectory.

        Parameters
        ----------
        condition : xarray.DataArray
            Boolean condition indicating where NaN values should be inserted.

        Returns
        -------
        xarray.Dataset
            Dataset with NaN values inserted at specified positions.
        """
        num_inserts = condition.sum(dim=self.obs_dim)
        max_obs = (self.index_of_last() + 1 + num_inserts).max().values

        # Create new empty dataset with extended obs dimension
        trajcoord = range(self.ds.sizes[self.trajectory_dim])
        nd = xr.Dataset(
            coords={
                self.trajectory_dim: ([self.trajectory_dim], trajcoord),
                self.obs_dim: ([self.obs_dim], range(max_obs))  # Longest trajectory
            },
            attrs=self.ds.attrs
        )

        # Add extended variables
        for varname, var in self.ds.data_vars.items():
            if self.obs_dim not in var.dims:
                nd[varname] = var
                continue

            # Create empty DataArray to hold interpolated values for the variable
            da = xr.DataArray(
                data=np.full((nd.sizes[self.trajectory_dim], nd.sizes[self.obs_dim]), np.nan),
                dims=nd.dims,
                attrs=var.attrs,
            )

            for t in range(self.ds.sizes[self.trajectory_dim]):  # Loop over trajectories
                numins = num_inserts[t]
                olddata = var.isel({self.trajectory_dim: t}).values
                wh = np.argwhere(condition.isel({self.trajectory_dim: t}).values) + 1
                if len(wh) == 0:
                    newdata = olddata
                else:
                    insert_indices = np.concatenate(wh)
                    s = np.split(olddata, insert_indices)

                    if np.issubdtype(var.dtype, np.datetime64):
                        na = np.atleast_1d(np.datetime64("NaT"))
                    else:
                        na = np.atleast_1d(np.nan)
                    newdata = np.concatenate(
                        [np.concatenate((ss, na)) for ss in s]
                    )

                newdata = newdata[:max_obs]  # Truncate to max_obs
                da[{self.trajectory_dim: t, self.obs_dim: slice(0, len(newdata))}] = newdata

            nd[varname] = da.astype(var.dtype)

        nd = nd.assign_coords({self.obs_dim: np.arange(max_obs)})  # New obs index 1..N

        return nd

    def drop_where(self, condition):
        """
        Remove positions where the condition is True, shifting the rest of the trajectory.

        Parameters
        ----------
        condition : xarray.DataArray
            Boolean condition indicating positions to drop.

        Returns
        -------
        xarray.Dataset
            Dataset with positions removed where the condition is True.
        """
        trajs = []
        newlen = 0
        for i in range(self.ds.sizes[self.trajectory_dim]):
            new = self.ds.isel({self.trajectory_dim: i}).drop_sel(obs=np.where(
                condition.isel({self.trajectory_dim: i}))[0])
            newlen = max(newlen, new.sizes[self.obs_dim])
            trajs.append(new)

        # Ensure all trajectories have equal length by padding with NaN at the end
        trajs = [
            t.pad(pad_width={self.obs_dim: (0, newlen - t.sizes[self.obs_dim])}).
                    assign_coords({self.obs_dim: np.arange(newlen)})  # New obs index 1..N
            for t in trajs
        ]

        ds = xr.concat(trajs, dim=self.trajectory_dim, join='exact')

        return ds

    @__require_obs_dim__
    def condense_obs(self) -> xr.Dataset:

        on = self.ds.sizes[self.obs_dim]
        logger.debug(f'Condensing {on} observations.')

        ds = self.ds.copy(deep=True)

        # The observation coordinate will be re-written
        ds = ds.drop_vars([self.obs_dim], errors='ignore')

        assert self.obs_dim in ds[
            self.
            time_varname].dims, "observation not a coordinate of time variable"

        # Move all observations for each trajectory to starting row
        maxN = 0
        for ti in range(len(ds.trajectory)):
            obsvars = [
                var for var in ds.variables if self.obs_dim in ds[var].dims
            ]
            iv = np.full(on, False)
            for var in obsvars:
                ivv = ~pd.isnull(ds[self.time_varname][
                    ti, :])  # valid times in this trajectory.
                iv = np.logical_or(iv, ivv)

            N = np.count_nonzero(iv)
            maxN = max(N, maxN)
            # logger.debug(f'Condensing trajectory {ti=}, observations: {N}..')

            if N == 0:
                logger.error(f'No valid observations in trajectory {ti}.')
                continue

            for var in obsvars:
                # logger.debug(f'Condensing variable {var}..')
                n = np.count_nonzero(~pd.isnull(ds[var][ti, :]))
                assert n <= N, f"Unexpected number of observations in trajectory for {ti=}, {var}: {n} > {N}."

                ds[var][ti, :N] = ds[var][ti, iv]
                ds[var][ti, N:] = np.nan

                # assert (~np.isnan(ds[var][ti, :N])).all(
                # ), "Varying number of valid observations within same trajectory."

        logger.debug(f'Condensed observations from: {on} to {maxN}')
        ds = ds.isel({self.obs_dim: slice(0, maxN)})

        # Write new observation coordinate.
        ds = ds.assign_coords({self.obs_dim: np.arange(0, maxN)})

        return ds

    def append(self, da, obs_dims=None):
        ds = self.ds.copy(deep=True)

        if obs_dims is None:
            obs_dims = [self.obs_dim]
        else:
            obs_dims = list(obs_dims)

        # Increase obs_dims to size of max
        for o in obs_dims:
            print(o)
            N = max(ds.sizes[o], da.sizes[o])
            ds = ds.pad({o: (0, N - ds.sizes[o])})
            da = da.pad({o: (0, N - da.sizes[o])})

        ds = xr.concat((ds, da), dim='trajectory')

        return ds

    def sel(self, *args, **kwargs):
        return self.ds.groupby(self.trajectory_dim).map(
            lambda d: ensure_time_dim(d.traj.to_1d().traj.sel(*args, **kwargs), self.time_varname).traj.to_2d(self.obs_dim))

    def seltime(self, t0=None, t1=None):
        # Using TrajAn sel method that allows NaN
        return self.ds.groupby(self.trajectory_dim).map(
            lambda d: ensure_time_dim(d.traj.to_1d().traj.seltime(t0, t1), self.time_varname).traj.to_2d(self.obs_dim))

    @__require_obs_dim__
    def iseltime(self, i):

        def select(t):
            ii = np.argwhere(~pd.isna(t[self.time_varname]).squeeze())
            ii = ii[i].squeeze() if len(ii) > 0 else ii.squeeze()

            o = t.isel({self.obs_dim: ii})

            if self.obs_dim in o.dims:
                return o
            else:
                return o.expand_dims(self.obs_dim)

        return self.ds.groupby(self.trajectory_dim).map(select)

    def to_1d(self):
        if self.ds.sizes[self.trajectory_dim] > 1:
            raise ValueError(
                "Can not convert a 2D dataset with multiple trajectories to 1D."
            )
        else:
            ds = self.ds.copy()

            # Do not remove NaN's since these now have meaning
            #ds = ds.dropna(self.obs_dim, how='all')

            # For 1D objects, we rename obs-dimension to name of time variable
            # so that time becomes a coordinate variable,
            # i.e. typically:   time(traj, obs) -> time(time)
            ds = ds.assign_coords({self.obs_dim: ds[self.time_varname]})
            ds = ds.drop_vars(self.time_varname).rename({self.obs_dim: self.time_varname})
            ds[self.time_varname] = ds[self.time_varname].squeeze(self.trajectory_dim)

            # Do not remove NaN's since these now have meaning
            #ds = ds.loc[{self.time_varname: ~pd.isna(ds[self.time_varname])}]
            #_, ui = np.unique(ds[self.time_varname], return_index=True)
            #ds = ds.isel({self.time_varname: ui})

            # Keep trajectory dimension, although always length 1 for 1D objects
            ds = ds.assign_coords({self.trajectory_dim: ds[self.trajectory_dim]})

            return ds

    @__require_obs_dim__
    def gridtime(self, *args, **kwargs):
        """
        Interpolate the dataset to a given time grid.

        Parameters
        ----------
        times : str, pandas.Timedelta, or numpy.ndarray
            Time grid to interpolate to. If a string or Timedelta, it specifies the interval.
        time_varname : str, optional
            Name of the time variable, by default the dataset's time variable.
        round : bool, optional
            Whether to round the start and end times to the nearest interval, by default True.

        Returns
        -------
        xarray.Dataset
            Dataset interpolated to the specified time grid.
        """
        gridded = self.ds.groupby(self.trajectory_dim).map(
                lambda d: ensure_time_dim(d.traj.to_1d().traj.gridtime(*args, **kwargs), self.time_varname))

        # TODO: trajectory index should be preserved so that this should not be necessary
        gridded = gridded.assign_coords({self.trajectory_dim: self.ds[self.trajectory_dim]})

        return gridded

    def skill(self):
        raise ValueError('Not implemented for 2D datasets')
