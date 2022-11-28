"""
Extending xarray Dataset with functionality specific to trajectory datasets.

Presently supporting Cf convention H.4.1
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_multidimensional_array_representation_of_trajectories.
"""

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import xarray as xr
import trajan as ta
import logging

from .plot import Plot

logger = logging.getLogger(__name__)

@xr.register_dataset_accessor("traj")
class TrajAccessor:
    _obj = None
    __plot__ = None

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
        # Identify dimension names
        for dim in ['obs', 'time']:
            if dim in self._obj.dims:
                self.timedim = dim
        for dim in ['traj', 'trajectory']:
            if dim in self._obj.dims:
                self.trajdim = dim
        if not hasattr(self, 'timedim'):
            raise ValueError(f'Time dimension not identified: {self._obj.dims}')

    @property
    def plot(self):
        if self.__plot__ is None:
            logger.debug(f'Setting up new plot object.')
            self.__plot__ = Plot(self._obj)

        return self.__plot__

    def gridtime(self, times):
        """Interpolate dataset to regular time interval

        times:
            - an array of times, or
            - a string "freq" specifying a Pandas daterange (e.g. 'h', '6h, 'D'...)

        Note that the resulting DataSet will have "time" as a dimension coordinate.
        """

        if isinstance(times, str):  # Make time series with given interval
            freq = times
            start_time = np.nanmin(np.asarray(self._obj.time))
            start_time = pd.to_datetime(start_time).strftime('%Y-%m-%d')
            end_time = np.nanmax(np.asarray(self._obj.time)) + \
                            np.timedelta64(23, 'h') + np.timedelta64(59, 'm')
            end_time = pd.to_datetime(end_time).strftime('%Y-%m-%d')
            times = pd.date_range(start_time, end_time, freq=freq)

        # Create empty dataset to hold interpolated values
        trajcoord = range(self._obj.dims['trajectory'])
        d = xr.Dataset(
            coords={
                'trajectory': (["trajectory"], trajcoord),
                'time': (["time"], times)
                    },
            attrs = self._obj.attrs
            )

        for varname, var in self._obj.variables.items():
            if varname in ['time', 'obs']:
                continue
            if self.timedim not in var.dims:
                d['varname'] = var
                continue

            # Create empty dataarray to hold interpolated values for given variable
            da = xr.DataArray(
                data=np.zeros(tuple(d.dims[di] for di in ['trajectory', 'time']))*np.nan,
                dims=d.dims,
                coords=d.coords,
                attrs=var.attrs
                )

            for t in range(self._obj.dims['trajectory']):  # loop over trajectories
                origtimes = self._obj['time'].isel(trajectory=t).astype(np.float64).values
                validtime = np.nonzero(~np.isnan(origtimes))[0]
                interptime = origtimes[validtime]
                interpvar = var.isel(trajectory=t).data
                # Make interpolator
                f = interp1d(interptime, interpvar, bounds_error=False)
                # Interpolate onto given times
                da.loc[{'trajectory': t}] = f(times.to_numpy().astype(np.float64))

            d[varname] = da

        return d
