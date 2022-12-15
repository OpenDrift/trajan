"""
Extending xarray Dataset with functionality specific to trajectory datasets.

Presently supporting Cf convention H.4.1
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_multidimensional_array_representation_of_trajectories.
"""

import numpy as np
import pandas as pd
import xarray as xr
import pyproj
import logging

from .plot import Plot
from .traj import Traj
from .traj1d import Traj1d
from .traj2d import Traj2d

logger = logging.getLogger(__name__)


@xr.register_dataset_accessor("traj")
class TrajAccessor:
    _ds: xr.Dataset
    __plot__: Plot
    inner: Traj

    def __init__(self, xarray_obj):
        self._ds = xarray_obj
        self.__plot__ = None

        if 'traj' in self.ds.dims:
            logger.info(
                'Normalizing dimension name from "traj" to "trajectory".')
            self._ds = self._ds.rename({'traj': 'trajectory'})

        if not 'trajectory' in self.ds.dims:
            raise ValueError(
                f'Trajectory dimension not identified: {self.ds.dims}')

        if len(self.ds['time'].shape) == 1:
            logger.debug('Detected structured (1D) trajectory dataset')
            self.inner = Traj1d(self._ds)
        elif len(self.ds['time'].shape) == 2:
            logger.debug('Detected un-structured (2D) trajectory dataset')
            self.inner = Traj2d(self._ds)
        else:
            raise ValueError(f'Time dimension has shape greater than 2: {self.ds["time"].shape}')


    @property
    def plot(self):
        if self.__plot__ is None:
            logger.debug(f'Setting up new plot object.')
            self.__plot__ = Plot(self.ds)

        return self.__plot__

    @property
    def ds(self):
        return self._ds

    def __getattr__(self, attr):
        """
        Forward all other method calls and attributes to traj instance.
        """
        return getattr(self.inner, attr)

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

    def distance_to_next(self):
        """Returns distance in m from one position to the next

           Last time is repeated for last position (which has no next position)
        """
        lon = self.ds.lon
        lat = self.ds.lat
        lenobs = self.ds.dims['obs']
        lonfrom = lon.isel(obs=slice(0, lenobs - 1))
        latfrom = lat.isel(obs=slice(0, lenobs - 1))
        lonto = lon.isel(obs=slice(1, lenobs))
        latto = lat.isel(obs=slice(1, lenobs))
        geod = pyproj.Geod(ellps='WGS84')
        azimuth_forward, a2, distance = geod.inv(lonfrom, latfrom, lonto,
                                                 latto)

        distance = xr.DataArray(distance, coords=lon.coords, dims=lon.dims)
        distance = xr.concat((distance, distance.isel(obs=-1)),
                             dim='obs')  # repeating last time step to
        return distance

    def speed(self):
        """Returns the speed [m/s] along trajectories"""

        distance = self.distance_to_next()
        timedelta_seconds = self.time_to_next() / np.timedelta64(1, 's')

        return distance / timedelta_seconds

    def index_of_last(self):
        """Find index of last valid position along each trajectory"""
        return np.ma.notmasked_edges(np.ma.masked_invalid(self.ds.lon.values),
                                     axis=1)[1][1]

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
