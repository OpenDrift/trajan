import logging
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

logger = logging.getLogger(__name__)


class Plot:
    ds = None
    ax = None

    # A lon-lat projection with the currently used globe.
    gcrs = None

    DEFAULT_LINE_COLOR = 'gray'

    def __init__(self, ds, ax=None):
        self.ds = ds
        self.ax = ax

    def set_up_map(self, crs=None):
        """
        Set up axes for plotting.

        Args:

            crs: Use a different crs than Mercator.

        Returns:

            An matplotlib axes with a Cartopy projection.

        """
        if self.ax is not None:
            return self.ax

        crs = crs if crs is not None else ccrs.Mercator()
        self.gcrs = ccrs.PlateCarree(globe=crs.globe)

        self.ax = plt.axes(projection=crs)

        gl = self.ax.gridlines(self.gcrs, draw_labels=True)
        gl.top_labels = None

        # TODO: Add landmask

        return self.ax

    def __call__(self, *args, **kwargs):
        return self.lines(*args, **kwargs)

    def lines(self, ax=None, crs=None, *args, **kwargs):
        """
        Plot the trajectory lines.

        Args:

            ax: Use existing axes, otherwise a new one is set up.

            crs: Specify crs for new axis.

        Returns:

            Matplotlib lines, and axes.
        """
        assert crs is None or ax is None, "Only one of `ax` and `crs` may be specified."

        logger.debug(f'Plotting lines {ax=}')
        ax = ax if ax is not None else self.set_up_map(crs=crs)

        if 'color' not in kwargs:
            kwargs['color'] = self.DEFAULT_LINE_COLOR

        paths = ax.plot(self.ds.lon.T,
                        self.ds.lat.T,
                        transform=self.gcrs,
                        *args,
                        **kwargs)

        return paths, ax
