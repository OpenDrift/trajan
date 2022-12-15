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
            if varname in ['time', 'obs']:
                continue
            if 'time' not in var.dims:
                d[varname] = var
                continue

            # Create empty dataarray to hold interpolated values for given variable
            da = xr.DataArray(data=np.zeros(
                tuple(d.dims[di] for di in ['trajectory', 'time'])) * np.nan,
                              dims=d.dims,
                              coords=d.coords,
                              attrs=var.attrs)

            for t in range(
                    self.ds.dims['trajectory']):  # loop over trajectories
                origtimes = self.ds['time'].isel(trajectory=t).astype(
                    np.float64).values
                validtime = np.nonzero(~np.isnan(origtimes))[0]
                interptime = origtimes[validtime]
                interpvar = var.isel(trajectory=t).data
                # Make interpolator
                f = interp1d(interptime, interpvar, bounds_error=False)
                # Interpolate onto given times
                da.loc[{
                    'trajectory': t
                }] = f(times.to_numpy().astype(np.float64))

            d[varname] = da

        return d
