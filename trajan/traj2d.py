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
        Return median time step between observations in seconds.
        """
        td = np.diff(self.ds.time, axis=1) / np.timedelta64(1, 's')
        td = average(td)
        return td

    def time_to_next(self):
        """Return time from one position to the next.

           Returned datatype is np.timedelta64
           Last time is repeated for last position (which has no next position)
        """
        time = self.ds.time
        lenobs = self.ds.sizes[self.obs_dim]
        td = time.isel(obs=slice(1, lenobs)) - time.isel(
            obs=slice(0, lenobs - 1))
        td = xr.concat((td, td.isel(obs=-1)),
                       dim=self.obs_dim)  # repeating last time step
        return td

    def is_1d(self):
        return False

    def is_2d(self):
        return True

    def insert_nan_where(self, condition):
        """Insert NaN-values in trajectories after given positions, shifting rest of trajectory."""

        index_of_last = self.index_of_last()
        num_inserts = condition.sum(dim=self.obs_dim)
        max_obs = (index_of_last + num_inserts).max().values

        # Create new empty dataset with extended obs dimension
        trajcoord = range(self.ds.sizes[self.trajectory_dim])
        nd = xr.Dataset(
            coords={
                self.trajectory_dim:
                ([self.trajectory_dim],
                 range(self.ds.sizes[self.trajectory_dim])),
                self.obs_dim:
                ([self.obs_dim], range(max_obs))  # Longest trajectory
            },
            attrs=self.ds.attrs)

        # Add extended variables
        for varname, var in self.ds.data_vars.items():
            if self.obs_dim not in var.dims:
                nd[varname] = var
                continue
            # Create empty dataarray to hold interpolated values for given variable
            da = xr.DataArray(
                data=np.zeros(tuple(nd.sizes[di] for di in nd.dims)) * np.nan,
                dims=nd.dims,
                attrs=var.attrs,
            )

            for t in range(self.ds.sizes[
                    self.trajectory_dim]):  # loop over trajectories
                numins = num_inserts[t]
                olddata = var.isel(trajectory=t).values
                wh = np.argwhere(condition.isel(trajectory=t).values) + 1
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
                        [np.concatenate((ss, na)) for ss in s])

                newdata = newdata[slice(0, max_obs -
                                        1)]  # truncating, should be checked
                da[{
                    self.trajectory_dim: t,
                    self.obs_dim: slice(0, len(newdata))
                }] = newdata

            nd[varname] = da.astype(var.dtype)

        nd = nd.drop_vars(
            (self.obs_dim, self.trajectory_dim))  # Remove coordinates

        return nd

    def drop_where(self, condition):
        """Remove positions where condition is True, shifting rest of trajectory."""

        trajs = []
        newlen = 0
        for i in range(self.ds.sizes[self.trajectory_dim]):
            new = self.ds.isel(trajectory=i).drop_sel(obs=np.where(
                condition.isel(
                    trajectory=i))[0])  # Dropping from given trajectory
            newlen = max(newlen, new.sizes[self.obs_dim])
            trajs.append(new)

        # Ensure all trajectories have equal length, by padding with NaN at end
        trajs = [
            t.pad(
                pad_width={self.obs_dim: (0, newlen - t.sizes[self.obs_dim])})
            for t in trajs
        ]

        return xr.concat(trajs, dim=self.trajectory_dim)

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
        obs = np.arange(0, maxN)
        ds = ds.assign_coords({self.obs_dim: obs})

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
            lambda d: ensure_time_dim(d.traj.to_1d().sel(*args, **kwargs), self
                                      .time_varname).traj.to_2d(self.obs_dim))

    def seltime(self, t0=None, t1=None):
        return self.sel({self.time_varname: slice(t0, t1)})

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
            ds = ds.dropna(self.obs_dim, how='all')
            ds = ds.assign_coords({self.obs_dim: ds[self.time_varname]})
            ds = ds.drop_vars(self.time_varname).rename(
                {self.obs_dim: self.time_varname})

            ds[self.time_varname] = ds[self.time_varname].squeeze(
                self.trajectory_dim)
            ds = ds.loc[{self.time_varname: ~pd.isna(ds[self.time_varname])}]
            _, ui = np.unique(ds[self.time_varname], return_index=True)
            ds = ds.isel({self.time_varname: ui})
            ds = ds.assign_coords(
                {self.trajectory_dim: ds[self.trajectory_dim]})

            return ds

    @__require_obs_dim__
    def gridtime(self, times, time_varname=None, round=True):
        if isinstance(times, str) or isinstance(
                times, pd.Timedelta):  # Make time series with given interval
            if round is True:
                start_time = np.nanmin(np.asarray(
                    self.ds.time.dt.floor(times)))
                end_time = np.nanmax(np.asarray(self.ds.time.dt.ceil(times)))
            else:
                start_time = np.nanmin(np.asarray(self.ds.time))
                end_time = np.nanmax(np.asarray(self.ds.time))
            times = pd.date_range(start_time,
                                  end_time,
                                  freq=times,
                                  inclusive='both')

        if not isinstance(times, np.ndarray):
            times = times.to_numpy()

        time_varname = self.time_varname if time_varname is None else time_varname

        d = None

        for t in range(self.ds.sizes[self.trajectory_dim]):
            dt = self.ds.isel({self.trajectory_dim : t}) \
                        .dropna(self.obs_dim, how='all')

            dt = dt.assign_coords({self.obs_dim : dt[self.time_varname].values }) \
                   .drop_vars(self.time_varname) \
                   .rename({self.obs_dim : time_varname}) \
                   .set_index({time_varname: time_varname})

            _, ui = np.unique(dt[time_varname], return_index=True)
            dt = dt.isel({time_varname: ui})
            dt = dt.isel(
                {time_varname: np.where(~pd.isna(dt[time_varname].values))[0]})

            if dt.sizes[time_varname] > 0:
                dt = dt.interp({time_varname: times})
            else:
                logger.warning(f"time dimension ({time_varname}) is zero size")

            if d is None:
                d = dt.expand_dims(self.trajectory_dim)
            else:
                d = xr.concat((d, dt), self.trajectory_dim)

        d = d.assign_coords(
            {self.trajectory_dim: self.ds[self.trajectory_dim]})

        return d

    def skill(self):
        raise ValueError('Not implemented for 1D datasets')
