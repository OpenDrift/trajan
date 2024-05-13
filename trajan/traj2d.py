import xarray as xr
import numpy as np
import pandas as pd
import logging

from .traj import Traj, __require_obsdim__

logger = logging.getLogger(__name__)


class Traj2d(Traj):
    """
    A unstructured dataset, where each trajectory may have observations at different times. Typically from a collection of drifters.
    """

    def __init__(self, ds, obsdim, timedim):
        super().__init__(ds, obsdim, timedim)

    def timestep(self, average=np.nanmedian):
        """
        Return median time step between observations in seconds.
        """
        td = np.diff(self.ds.time, axis=1) / np.timedelta64(1, 's')
        td = average(td)
        return td

    def time_to_next(self):
        """Returns time from one position to the next

           Returned datatype is np.timedelta64
           Last time is repeated for last position (which has no next position)
        """
        time = self.ds.time
        lenobs = self.ds.sizes['obs']
        td = time.isel(obs=slice(1, lenobs)) - time.isel(
            obs=slice(0, lenobs - 1))
        td = xr.concat((td, td.isel(obs=-1)),
                       dim='obs')  # repeating last time step
        return td

    def is_1d(self):
        return False

    def is_2d(self):
        return True

    def insert_nan_where(self, condition):
        """Insert NaN-values in trajectories after given positions, shifting rest of trajectory"""

        index_of_last = self.index_of_last()
        num_inserts = condition.sum(dim='obs')
        max_obs = (index_of_last + num_inserts).max().values

        # Create new empty dataset with extended obs dimension
        trajcoord = range(self.ds.sizes['trajectory'])
        nd = xr.Dataset(
            coords={
                'trajectory':
                (["trajectory"], range(self.ds.sizes['trajectory'])),
                'obs': (['obs'], range(max_obs))  # Longest trajectory
            },
            attrs=self.ds.attrs)

        # Add extended variables
        for varname, var in self.ds.data_vars.items():
            if 'obs' not in var.dims:
                nd[varname] = var
                continue
            # Create empty dataarray to hold interpolated values for given variable
            da = xr.DataArray(
                data=np.zeros(tuple(nd.sizes[di] for di in nd.dims)) * np.nan,
                dims=nd.dims,
                attrs=var.attrs,
            )

            for t in range(
                    self.ds.sizes['trajectory']):  # loop over trajectories
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
                da[{'trajectory': t, 'obs': slice(0, len(newdata))}] = newdata

            nd[varname] = da.astype(var.dtype)

        nd = nd.drop_vars(('obs', 'trajectory'))  # Remove coordinates

        return nd

    def drop_where(self, condition):
        """Remove positions where condition is True, shifting rest of trajectory"""

        trajs = []
        newlen = 0
        for i in range(self.ds.sizes['trajectory']):
            new = self.ds.isel(trajectory=i).drop_sel(obs=np.where(
                condition.isel(
                    trajectory=i))[0])  # Dropping from given trajectory
            newlen = max(newlen, new.sizes['obs'])
            trajs.append(new)

        # Ensure all trajectories have equal length, by padding with NaN at end
        trajs = [
            t.pad(pad_width={'obs': (0, newlen - t.sizes['obs'])})
            for t in trajs
        ]

        return xr.concat(trajs, dim='trajectory')

    @__require_obsdim__
    def condense_obs(self):
        """
        Move all observations to the first index, so that the observation
        dimension is reduced to a minimum. When creating ragged arrays the
        observations from consecutive trajectories start at the observation
        index after the previous, causing a very long observation dimension.

        Original:

        .............. Observations --->
        trajectory 1: | t01 | t02 | t03 | t04 | t05 | nan | nan | nan | nan |
        trajectory 2: | nan | nan | nan | nan | nan | t11 | t12 | t13 | t14 |

        After condensing:

        .............. Observations --->
        trajectory 1: | t01 | t02 | t03 | t04 | t05 |
        trajectory 2: | t11 | t12 | t13 | t14 | nan |

        Returns:

            A new Dataset with observations condensed.
        """

        on = self.ds.sizes[self.obsdim]
        logger.debug(f'Condensing {on} observations.')

        ds = self.ds.copy(deep=True)

        # The observation coordinate will be re-written
        ds = ds.drop_vars([self.obsdim])

        assert self.obsdim in ds[
            self.timedim].dims, "observation not a coordinate of time variable"

        # Move all observations for each trajectory to starting row
        maxN = 0
        for ti in range(len(ds.trajectory)):
            obsvars = [
                var for var in ds.variables if self.obsdim in ds[var].dims
            ]
            iv = np.full(on, False)
            for var in obsvars:
                ivv = ~pd.isnull(
                    ds[self.timedim][ti, :])  # valid times in this trajectory.
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
        ds = ds.isel({self.obsdim: slice(0, maxN)})

        # Write new observation coordinate.
        obs = np.arange(0, maxN)
        ds = ds.assign_coords({self.obsdim: obs})

        return ds

    @__require_obsdim__
    def gridtime(self, times, timedim=None, round=True):
        if isinstance(times, str) or isinstance(times, pd.Timedelta):  # Make time series with given interval
            if round is True:
                start_time = np.nanmin(np.asarray(self.ds.time.dt.floor(times)))
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

        timedim = self.timedim if timedim is None else timedim

        d = None

        for t in range(self.ds.sizes['trajectory']):
            dt = self.ds.isel(trajectory=t) \
                        .dropna(self.obsdim, how='all')

            dt = dt.assign_coords({self.obsdim : dt[self.timedim].values }) \
                   .drop_vars(self.timedim) \
                   .rename({self.obsdim : timedim}) \
                   .set_index({timedim: timedim})

            _, ui = np.unique(dt[timedim], return_index=True)
            dt = dt.isel({timedim: ui})
            dt = dt.isel({timedim : np.where(~pd.isna(dt[timedim].values))[0]})

            if dt.sizes[timedim] > 0:
                dt = dt.interp({timedim: times})
            else:
                logger.warning(f"time dimension ({timedim}) is zero size")

            if d is None:
                d = dt.expand_dims('trajectory')
            else:
                d = xr.concat((d, dt), "trajectory")

        d = d.assign_coords({'trajectory': self.ds['trajectory']})

        return d
