import xarray as xr
import logging
import numpy as np

from .plot import Plot

# recommended by cf-xarray
xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)


@xr.register_dataarray_accessor('wave')
class Wave:
    def __init__(self, ds):
        self.ds = ds
        self.__plot__ = None

    @property
    def plot(self) -> Plot:
        """
        See :class:`trajan.waves.Plot`.
        """
        if self.__plot__ is None:
            logger.debug(f'Setting up new plot object.')
            self.__plot__ = Plot(self.ds)

        return self.__plot__
