"""
Plot land shapes and landmask.
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import logging

logger = logging.getLogger(__name__)

__mask__ = None


def __get_mask__():
    """
    Return an instance of the global landmask.
    """
    global __mask__

    try:
        # Use the cached landmask from OpenDrift if it exists
        from opendrift.readers import reader_global_landmask
        if reader_global_landmask.__roaring_mask__ is not None:
            logger.debug('Using roaring landmask instance from OpenDrift.')
            __mask__ = reader_global_landmask.__roaring_mask__
    except ImportError:
        pass

    if __mask__ is None:
        logger.debug('Setting up roaring landmask..')
        from roaring_landmask import RoaringLandmask
        __mask__ = RoaringLandmask.new()

    return __mask__


def add_land(ax,
             lonmin,
             latmin,
             lonmax,
             latmax,
             fast,
             ocean_color='white',
             land_color=cfeature.COLORS['land'],
             lscale='auto',
             globe=None):
    """
    Plot the landmask or the shapes from GSHHG.
    """

    def show_landmask_roaring(roaring):
        maxn = 512.
        dx = (lonmax - lonmin) / maxn
        dy = (latmax - latmin) / maxn
        dx = max(roaring.dx, dx)
        dy = max(roaring.dy, dy)

        x = np.arange(lonmin, lonmax, dx)
        y = np.arange(latmin, latmax, dy)

        yy, xx = np.meshgrid(y, x)
        img = roaring.mask.contains_many_par(xx.ravel(),
                                             yy.ravel()).reshape(yy.shape).T

        from matplotlib import colors
        cmap = colors.ListedColormap([ocean_color, land_color])
        ax.imshow(img,
                  origin='lower',
                  extent=[lonmin.values, lonmax.values, latmin.values, latmax.values],
                  zorder=0,
                  transform=ccrs.PlateCarree(globe=globe),
                  cmap=cmap)

    if fast:
        show_landmask_roaring(__get_mask__())
    else:
        land = cfeature.GSHHSFeature(scale=lscale, facecolor=land_color)

        ax.add_feature(land, zorder=2, facecolor=land_color, edgecolor='black')

