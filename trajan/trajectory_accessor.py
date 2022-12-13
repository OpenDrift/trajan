"""
Extending xarray Dataset with functionality specific to trajectory datasets.

Presently supporting Cf convention H.4.1
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_multidimensional_array_representation_of_trajectories.
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import pyproj
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
                d[varname] = var
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

    def time_to_next(self):
        """Returns time from one position to the next

           Returned datatype is np.timedelta64
           Last time is repeated for last position (which has no next position)
        """
        time = self._obj.time
        lenobs = self._obj.dims['obs']
        td = time.isel(obs=slice(1, lenobs))-time.isel(obs=slice(0, lenobs-1))
        td = xr.concat((td, td.isel(obs=-1)), dim='obs')  # repeating last time step
        return td

    def distance_to_next(self):
        """Returns distance in m from one position to the next

           Last time is repeated for last position (which has no next position)
        """
        lon = self._obj.lon
        lat = self._obj.lat
        lenobs = self._obj.dims['obs']
        lonfrom = lon.isel(obs=slice(0, lenobs-1))
        latfrom = lat.isel(obs=slice(0, lenobs-1))
        lonto = lon.isel(obs=slice(1, lenobs))
        latto = lat.isel(obs=slice(1, lenobs))
        geod = pyproj.Geod(ellps='WGS84')
        azimuth_forward, a2, distance = geod.inv(lonfrom, latfrom, lonto, latto)

        distance = xr.DataArray(distance, coords=lon.coords, dims=lon.dims)
        distance = xr.concat((distance, distance.isel(obs=-1)), dim='obs')  # repeating last time step to 
        return distance

    def speed(self):
        """Returns the speed [m/s] along trajectories"""

        distance = self.distance_to_next()
        timedelta_seconds = self.time_to_next()/np.timedelta64(1, 's')

        return distance / timedelta_seconds

    def index_of_last(self):
        """Find index of last valid position along each trajectory"""
        return np.ma.notmasked_edges(np.ma.masked_invalid(self._obj.lon.values), axis=1)[1][1]

    def insert_nan_where(self, condition):
        """Insert NaN-values in trajectories at given positions, shifting rest of trajectory"""

        index_of_last = self.index_of_last()
        num_inserts = condition.sum(dim='obs')
        max_obs = (index_of_last + num_inserts).max().values

        # Create new empty dataset with extended obs dimension
        trajcoord = range(self._obj.dims['trajectory'])
        nd = xr.Dataset(
            coords={
                'trajectory': (["trajectory"], range(self._obj.dims['trajectory'])),
                'obs': (["obs"], range(max_obs))  # Longest trajectory
                    },
            attrs = self._obj.attrs
            )

        # Add extended variables
        for varname, var in self._obj.data_vars.items():
            if self.timedim not in var.dims:
                nd[varname] = var
                continue
            # Create empty dataarray to hold interpolated values for given variable
            da = xr.DataArray(
                data=np.zeros(tuple(nd.dims[di] for di in nd.dims))*np.nan,
                dims=nd.dims,
                attrs=var.attrs,
                )

            for t in range(self._obj.dims['trajectory']):  # loop over trajectories
                numins = num_inserts[t]
                olddata = var.isel(trajectory=t).values
                wh = np.argwhere(condition.isel(trajectory=t).values)
                if len(wh) == 0:
                    newdata = olddata
                else:
                    insert_indices = np.concatenate(wh)
                    s = np.split(olddata, insert_indices)

                    if np.issubdtype(var.dtype, np.datetime64):
                        na = np.atleast_1d(np.datetime64("NaT"))
                    else:
                        na = np.atleast_1d(np.nan)
                    newdata = np.concatenate([np.concatenate((ss, na)) for ss in s])

                newdata = newdata[slice(0, max_obs-1)]  # truncating, should be checked
                da[{'trajectory': t, 'obs': slice(0, len(newdata))}] = newdata

            nd[varname] = da.astype(var.dtype)

        nd = nd.drop_vars(('obs', 'trajectory'))  # Remove coordinates

        return nd

    def drop_where(self, condition):
        """Remove positions where condition is True, shifting rest of trajectory"""

        trajs = []
        for i in range(self._obj.dims['trajectory']):
            new = self._obj.isel(trajectory=i).drop_sel(
                obs=np.where(condition.isel(trajectory=i))[0])  # Dropping from given trajectory
            new = new.pad(pad_width={'obs': (0, self._obj.dims['obs']-new.dims['obs'])}) # Pad with NaN
            trajs.append(new)

        return xr.concat(trajs, dim='trajectory')
