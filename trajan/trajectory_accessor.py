"""
Extending xarray Dataset with functionality specific to trajectory datasets.

Presently supporting Cf convention H.4.1
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_multidimensional_array_representation_of_trajectories.
"""

import numpy as np
from scipy.interpolate import interp1d
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


    @property
    def plot(self):
        if self.__plot__ is None:
            logger.debug(f'Setting up new plot object.')
            self.__plot__ = Plot(self._obj)

        return self.__plot__

    def gridtime(self, times):
        """Interpolate dataset to regular time interval"""

        # Create empty dataset to hold interpolated values
        d = xr.Dataset(
            coords={
                'time': (["obs"], times),
                'trajectory': (["trajectory"], self._obj['drifter_names'].data)
                    },
            attrs = self._obj.attrs
            )

        for varname, var in self._obj.variables.items():
            if varname == 'time':
                continue
            if var.dtype != np.float64:  # Copy without interpolation
                d['varname'] = var
                continue

            # Create empty dataarray to hold interpolated values for given variable
            da = xr.DataArray(
                data=np.zeros(tuple(d.dims[di] for di in ['trajectory', 'obs']))*np.nan,
                dims=self._obj.dims,
                coords=self._obj.coords,
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
