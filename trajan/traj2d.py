import pyproj
import xarray as xr
import numpy as np

from .traj import Traj

class Traj2d(Traj):
    """
    A unstructured dataset, where each trajectory may have observations at different times. Typically from a collection of drifters.
    """

    def __init__(self, ds):
        super().__init__(ds)

    def gridtime(self, times):
        """Interpolate dataset to regular time interval

        times:
            - an array of times, or
            - a string "freq" specifying a Pandas daterange (e.g. 'h', '6h, 'D'...)

        Note that the resulting DataSet will have "time" as a dimension coordinate.
        """

        from scipy.interpolate import interp1d
        import pandas as pd

        if isinstance(times, str):  # Make time series with given interval
            freq = times
            start_time = np.nanmin(np.asarray(self.ds.time))
            start_time = pd.to_datetime(start_time).strftime('%Y-%m-%d')
            end_time = np.nanmax(np.asarray(self.ds.time)) + \
                            np.timedelta64(23, 'h') + np.timedelta64(59, 'm')
            end_time = pd.to_datetime(end_time).strftime('%Y-%m-%d')
            times = pd.date_range(start_time, end_time, freq=freq)

        # Create empty dataset to hold interpolated values
        trajcoord = range(self.ds.dims['trajectory'])
        d = xr.Dataset(coords={
            'trajectory': (["trajectory"], trajcoord),
            'time': (["time"], times)
        },
                       attrs=self.ds.attrs)

        for varname, var in self.ds.variables.items():
            if varname == 'time':
                continue
            if 'obs' not in var.dims:
                d[varname] = var
                continue

            # Create empty dataarray to hold interpolated values for given variable
            da = xr.DataArray(data=np.zeros(
                tuple(d.dims[di] for di in ['trajectory', 'time'])) * np.nan,
                              dims=d.dims,
                              coords=d.coords,
                              attrs=var.attrs)

            origtimes = self.ds['time'].ffill(dim='obs').astype(np.float64)

            for t in range(
                    self.ds.dims['trajectory']):  # loop over trajectories
                # Make interpolator
                f = interp1d(origtimes.isel(trajectory=t), var.isel(trajectory=t), bounds_error=False)
                # Interpolate onto given times
                da.loc[{'trajectory': t}] = f(times.to_numpy().astype(np.float64))

            d[varname] = da

        return d

    def time_to_next(self):
        """Returns time from one position to the next

           Returned datatype is np.timedelta64
           Last time is repeated for last position (which has no next position)
        """
        time = self.ds.time
        lenobs = self.ds.dims['obs']
        td = time.isel(obs=slice(1, lenobs)) - time.isel(
            obs=slice(0, lenobs - 1))
        td = xr.concat((td, td.isel(obs=-1)),
                       dim='obs')  # repeating last time step
        return td

    def insert_nan_where(self, condition):
        """Insert NaN-values in trajectories after given positions, shifting rest of trajectory"""

        index_of_last = self.index_of_last()
        num_inserts = condition.sum(dim='obs')
        max_obs = (index_of_last + num_inserts).max().values

        # Create new empty dataset with extended obs dimension
        trajcoord = range(self.ds.dims['trajectory'])
        nd = xr.Dataset(
            coords={
                'trajectory': (["trajectory"], range(self.ds.dims['trajectory'])),
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
                data=np.zeros(tuple(nd.dims[di] for di in nd.dims)) * np.nan,
                dims=nd.dims,
                attrs=var.attrs,
            )

            for t in range(
                    self.ds.dims['trajectory']):  # loop over trajectories
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
        for i in range(self.ds.dims['trajectory']):
            new = self.ds.isel(trajectory=i).drop_sel(obs=np.where(
                condition.isel(
                    trajectory=i))[0])  # Dropping from given trajectory
            newlen = max(newlen, new.dims['obs'])
            trajs.append(new)

        # Ensure all trajectories have equal length, by padding with NaN at end
        trajs = [
            t.pad(pad_width={'obs': (0, newlen - t.dims['obs'])})
            for t in trajs
        ]

        return xr.concat(trajs, dim='trajectory')

