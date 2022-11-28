import logging
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

from .land import add_land

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True


class Plot:
    ds: xr.Dataset

    # A lon-lat projection with the currently used globe.
    gcrs = None

    DEFAULT_LINE_COLOR = 'gray'

    def __init__(self, ds):
        self.ds = ds
        self.gcrs = ccrs.PlateCarree()

    def set_up_map(
        self,
        kwargs_d=None,
        **kwargs
    ):
        """
        Set up axes for plotting.

        Args:

            crs: Use a different crs than Mercator.

            margin: margin (decimal degrees) in addition to extent of trajectories.

            land: Add land shapes based on GSHHG to map.

                'auto' (default): use automatic scaling.

                'c', 'l','i','h','f' or
                'coarse', 'low', 'intermediate', 'high', 'full': use corresponding GSHHG level.

                'mask' or 'fast' (fastest): use a raster mask generated from GSHHG.

                None: do not add land shapes.

        Returns:

            An matplotlib axes with a Cartopy projection.

        """
        # By popping the args from kwargs they are not passed onto matplotlib later.
        if kwargs_d is None:
            kwargs_d = kwargs
        else:
            for k,v in kwargs:
                kwargs_d[k] = v

        ax = kwargs_d.pop('ax', None)
        crs = kwargs_d.pop('crs', None)
        margin = kwargs_d.pop('margin', .1)
        corners = kwargs_d.pop('corners', None)
        land = kwargs_d.pop('land', 'auto')
        figsize = kwargs_d.pop('figsize', 11)

        assert crs is None or ax is None, "Only one of `ax` and `crs` may be specified."

        if ax is not None:
            logger.debug('axes already set up')
            return ax

        # It is not possible to change the projection of existing axes. The type of axes object returned
        # by `plt.axes` depends on the input projection.

        # Create a new figure if none exists.
        if corners is None:
            lonmin = self.ds.lon.min() - margin
            lonmax = self.ds.lon.max() + margin
            latmin = self.ds.lat.min() - margin
            latmax = self.ds.lat.max() + margin
        else:
            lonmin = corners[0]
            lonmax = corners[1]
            latmin = corners[2]
            latmax = corners[3]

        if len(plt.get_fignums()) == 0:
            logger.debug('Creating new figure and axes..')
            meanlat = (latmin + latmax) / 2
            aspect_ratio = float(latmax - latmin) / (float(lonmax - lonmin))
            aspect_ratio = aspect_ratio / np.cos(np.radians(meanlat))

            if aspect_ratio > 1:
                fig = plt.figure(figsize=(figsize / aspect_ratio, figsize))
            else:
                fig = plt.figure(figsize=(figsize, figsize * aspect_ratio))

            # fig.canvas.draw()  # maybe needed?
            plt.tight_layout()

        else:
            fig = plt.gcf()
            if len(fig.axes) > 0:
                logger.debug('Axes already exist on existing figure.')
                return fig.gca()
            else:
                logger.debug('Figure exists, setting up axes.')

        crs = crs if crs is not None else ccrs.Mercator()
        self.gcrs = ccrs.PlateCarree(globe=crs.globe)

        ax = fig.add_subplot(111, projection=crs)
        ax.set_extent([lonmin, lonmax, latmin, latmax], crs=self.gcrs)

        gl = ax.gridlines(self.gcrs, draw_labels=True)
        gl.top_labels = None

        add_land(ax, lonmin, latmin, lonmax, latmax,
                 fast=(land == 'mask' or land == 'fast'), lscale=land, globe=crs.globe)

        return ax

    def __call__(self, *args, **kwargs):
        return self.lines(*args, **kwargs)

    def lines(self, *args, **kwargs):
        """
        Plot the trajectory lines.

        Args:

            ax: Use existing axes, otherwise a new one is set up.

            crs: Specify crs for new axis.

        Returns:

            Matplotlib lines, and axes.
        """
        logger.debug(f'Plotting lines')
        ax = self.set_up_map(kwargs)

        if 'color' not in kwargs:
            kwargs['color'] = self.DEFAULT_LINE_COLOR

        if 'alpha' not in kwargs and 'trajectory' in self.ds.dims:
            num = self.ds.dims['trajectory']
            if num>100:  # If many trajectories, make more transparent
                kwargs['alpha'] = np.maximum(.1, 100/np.float64(num))

        paths = ax.plot(self.ds.lon.T,
                        self.ds.lat.T,
                        transform=self.gcrs,
                        *args,
                        **kwargs)

        return paths
