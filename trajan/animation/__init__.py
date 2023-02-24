import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from trajan.plot import Plot

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True

class Animation:
    ds: xr.Dataset

    # A lon-lat projection with the currently used globe.
    gcrs = None

    DEFAULT_LINE_COLOR = Plot.DEFAULT_LINE_COLOR

    def __init__(self, ds):
        self.ds = ds
        self.gcrs = ccrs.PlateCarree()

    def __call__(self, *args, **kwargs):
        return self.animate(*args, **kwargs)

    def animate(self, *args, **kwargs):
        logger.debug(f'Animating trajectories..')
        ax = self.ds.traj.plot.set_up_map(kwargs)
