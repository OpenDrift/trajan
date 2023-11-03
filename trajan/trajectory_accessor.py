"""
Extending xarray Dataset with functionality specific to trajectory datasets.

Presently supporting Cf convention H.4.1
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_multidimensional_array_representation_of_trajectories.
"""

import xarray as xr
import logging

from .plot import Plot
from .animation import Animation
from .traj import Traj, detect_tx_dim
from .traj1d import Traj1d
from .traj2d import Traj2d

logger = logging.getLogger(__name__)


@xr.register_dataset_accessor("traj")
class TrajAccessor:
    _ds: xr.Dataset
    __plot__: Plot
    __animate__: Animation
    inner: Traj

    def __init__(self, xarray_obj):
        self._ds = xarray_obj
        self.__plot__ = None
        self.__animate__ = None

        if 'traj' in self.ds.dims:
            logger.info(
                'Normalizing dimension name from "traj" to "trajectory".')
            self._ds = self._ds.rename({'traj': 'trajectory'})

        if 'trajectory' not in self.ds.dims:  # Add empty trajectory dimension, if single trajectory
            self._ds = self._ds.expand_dims({'trajectory': 1})
            self._ds['trajectory'].attrs['cf_role'] = 'trajectory_id'


        obsdim = None
        timedim = None

        tx = detect_tx_dim(self._ds)

        if 'obs' in tx.dims:
            obsdim = 'obs'
            timedim = self.__detect_time_dim__(obsdim)

        elif 'time' in tx.dims:
            obsdim = 'time'
            timedim = 'time'

        else:
            for d in tx.dims:
                if not self.ds[d].attrs.get('cf_role', None) == 'trajectory_id' and not 'traj' in d:

                    obsdim = d
                    timedim = self.__detect_time_dim__(obsdim)

                    break

            if obsdim is None:
                logger.warning('No time or obs dimension detected.')

        logger.debug(f"Detected obs-dim: {obsdim}, detected time-dim: {timedim}.")

        if obsdim is None:
            self.inner = Traj1d(self.ds, obsdim, timedim)

        elif len(self._ds[timedim].shape) <= 1:
            logger.debug('Detected structured (1D) trajectory dataset')
            self.inner = Traj1d(self._ds, obsdim, timedim)

        elif len(self._ds[timedim].shape) == 2:
            logger.debug('Detected un-structured (2D) trajectory dataset')
            self.inner = Traj2d(self._ds, obsdim, timedim)

        else:
            raise ValueError(f'Time dimension has shape greater than 2: {self.ds["timedim"].shape}')

    def __detect_time_dim__(self, obsdim):
        logger.debug(f'Detecting time-dimension for "{obsdim}"..')
        for v in self._ds.variables:
            if obsdim in self._ds[v].dims and 'time' in v:
                return v

        raise ValueError("no time dimension detected")

    @property
    def plot(self):
        if self.__plot__ is None:
            logger.debug(f'Setting up new plot object.')
            self.__plot__ = Plot(self.ds)

        return self.__plot__

    @property
    def animate(self):
        if self.__animate__ is None:
            logger.debug(f'Setting up new animation object.')
            self.__animate__ = Animation(self.ds)

        return self.__animate__

    @property
    def ds(self):
        return self._ds

    def __getattr__(self, attr):
        """
        Forward all other method calls and attributes to traj instance.
        """
        return getattr(self.inner, attr)
