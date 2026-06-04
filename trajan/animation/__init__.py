import logging
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes
import numpy as np
import xarray as xr

from trajan.plot import Plot

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True


class Animation:
    """
    Builder for trajectory animations.

    Obtain via ``ds.traj.animate``, configure with method chaining, then call
    :meth:`show` or :meth:`save` to render.

    Examples
    --------
    Simple:

    >>> ds.traj.animate().show()  # doctest: +SKIP

    Chained:

    >>> (ds.traj.animate()  # doctest: +SKIP
    ...     .color_by('speed', cmap='RdYlBu_r')
    ...     .show_trajectories(alpha=0.2)
    ...     .save('barents.mp4'))

    With background field:

    >>> (ds.traj.animate()  # doctest: +SKIP
    ...     .overlay_variable(current_speed_da, cmap='Blues', label='speed [m/s]')
    ...     .show())
    """

    ds: xr.Dataset
    gcrs = None

    DEFAULT_LINE_COLOR = Plot.DEFAULT_LINE_COLOR

    def __init__(self, ds):
        self.ds = ds
        self.gcrs = ccrs.PlateCarree()
        self._color = None
        self._cmap = None
        self._vmin = None
        self._vmax = None
        self._clabel = None
        self._colorbar = True
        self._show_trajectories = False
        self._trajectory_alpha = 0.1
        self._markersize = 20
        self._alpha = 1.0
        self._fps = 8
        self._title = 'auto'
        self._map_kwargs = {}
        self._overlays = []
        self._animation = None

    def __call__(self, **map_kwargs):
        """
        Configure map options and return ``self`` for chaining.

        Keyword arguments are forwarded to
        :meth:`trajan.plot.Plot.set_up_map` (e.g. ``ax``, ``crs``,
        ``land``, ``margin``, ``corners``).

        Returns
        -------
        self
        """
        self._map_kwargs.update(map_kwargs)
        return self

    # -- Configuration methods ------------------------------------------------

    def color_by(self, variable, cmap=None, vmin=None, vmax=None, label=None,
                 colorbar=True):
        """
        Color particles by a variable or a fixed matplotlib colour.

        Parameters
        ----------
        variable : str or xarray.DataArray
            A DataArray with dims ``(trajectory, time)``, a dataset variable
            name (e.g. ``'speed'``), or a fixed matplotlib colour string
            (e.g. ``'red'``).
        cmap : str or Colormap, optional
            Colormap (default: ``'jet'``).
        vmin, vmax : float, optional
            Colour scale limits.
        label : str, optional
            Colorbar label.
        colorbar : bool, optional
            Whether to draw a colorbar (default: ``True``).

        Returns
        -------
        self
        """
        self._color = variable
        if cmap is not None:
            self._cmap = cmap
        if vmin is not None:
            self._vmin = vmin
        if vmax is not None:
            self._vmax = vmax
        if label is not None:
            self._clabel = label
        self._colorbar = colorbar
        return self

    def show_trajectories(self, alpha=0.1):
        """
        Draw full trajectory lines as a static background.

        Parameters
        ----------
        alpha : float, optional
            Transparency (default: ``0.1``).

        Returns
        -------
        self
        """
        self._show_trajectories = True
        self._trajectory_alpha = alpha
        return self

    def set_fps(self, fps):
        """
        Set the playback speed.

        Parameters
        ----------
        fps : int

        Returns
        -------
        self
        """
        self._fps = fps
        self._animation = None
        return self

    def set_markersize(self, size):
        """
        Set the particle marker size.

        Parameters
        ----------
        size : float

        Returns
        -------
        self
        """
        self._markersize = size
        return self

    def set_title(self, title):
        """
        Control the axes title.

        Parameters
        ----------
        title : str or None
            ``'auto'`` (default) shows the current time stamp per frame inside
            the axes. ``None`` suppresses the timestamp entirely. Any other
            string is used as a fixed axes title (timestamp still shown inside).

        Returns
        -------
        self
        """
        self._title = title
        return self

    def overlay_variable(self, data, cmap='viridis', alpha=0.5, vmin=None,
                         vmax=None, label=None):
        """
        Overlay an animated gridded variable as a background field.

        Parameters
        ----------
        data : xarray.DataArray
            Must have a ``time`` dimension and ``lat``/``lon`` (or
            ``latitude``/``longitude``) coordinates.
        cmap : str or Colormap, optional
        alpha : float, optional
        vmin, vmax : float, optional
        label : str, optional
            Colorbar label.

        Returns
        -------
        self
        """
        self._overlays.append(
            dict(data=data, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                 label=label))
        return self

    # -- Rendering ------------------------------------------------------------

    def build(self, blit=True):
        """
        Build the :class:`matplotlib.animation.FuncAnimation`.

        The result is cached (for ``blit=True`` only); call :meth:`set_fps`
        to invalidate the cache and rebuild on the next call.

        Parameters
        ----------
        blit : bool, optional
            Use blitting for faster rendering (default ``True``).  Pass
            ``False`` when the output will be embedded as jshtml (Jupyter) to
            avoid artefacts caused by the blit background-capture path.

        Returns
        -------
        matplotlib.animation.FuncAnimation
        """
        if blit and self._animation is not None:
            return self._animation

        map_kwargs = dict(self._map_kwargs)  # copy so set_up_map can pop from it
        ax = self.ds.traj.plot.set_up_map(map_kwargs)
        fig = ax.get_figure()

        if self.ds.traj.is_1d():
            ds = self.ds
            logger.debug('Dataset is already regularly gridded, skipping gridtime')
        else:
            timestep = self.ds.traj.timestep()
            ds = self.ds.traj.gridtime(timestep)
            logger.debug(f'Gridded dataset to {timestep}')
        times = ds.time.values   # 1-D array after gridtime
        frames = len(times)
        logger.debug(f'Building animation: {frames} frames')

        is_cartesian = self.ds.traj.crs is None
        is_geo = (not is_cartesian
                  and isinstance(ax, cartopy.mpl.geoaxes.GeoAxes))

        if is_cartesian:
            x = ds.traj.tx.values   # (trajectory, time)
            y = ds.traj.ty.values
        else:
            x = ds.traj.tlon.values  # (trajectory, time)
            y = ds.traj.tlat.values

        # Static trajectory lines in the background
        if self._show_trajectories:
            self.ds.traj.plot.lines(ax=ax, alpha=self._trajectory_alpha)

        # -- Overlay variables (animated pcolormesh) --------------------------
        overlay_artists = []
        for ov in self._overlays:
            data = ov['data']
            lons = data.cf['longitude']
            lats = data.cf['latitude']
            ov_vmin = (ov['vmin'] if ov['vmin'] is not None
                       else float(np.nanmin(data.values)))
            ov_vmax = (ov['vmax'] if ov['vmax'] is not None
                       else float(np.nanmax(data.values)))
            data_interp = data.interp(time=times, method='nearest')
            pm_kwargs = dict(cmap=ov['cmap'], alpha=ov['alpha'],
                             vmin=ov_vmin, vmax=ov_vmax)
            if is_geo:
                pm_kwargs['transform'] = self.gcrs
            pm = ax.pcolormesh(lons, lats,
                               data_interp.isel(time=0).values, **pm_kwargs)
            if ov['label'] is not None:
                fig.colorbar(pm, ax=ax, label=ov['label'],
                             orientation='horizontal', pad=0.05, shrink=0.8)
            overlay_artists.append((pm, data_interp))

        # -- Particle scatter --------------------------------------------------
        cmap_obj = self._cmap if self._cmap is not None else 'jet'
        if isinstance(cmap_obj, str):
            cmap_obj = matplotlib.colormaps[cmap_obj]

        # Resolve the colour source: DataArray > variable name in ds > fixed colour
        colorarray = None
        vmin, vmax = self._vmin, self._vmax

        if isinstance(self._color, xr.DataArray):
            colorarray = self._color.values  # (trajectory, time)
            if self._clabel is None and self._color.name:
                self._clabel = self._color.name
        elif isinstance(self._color, str) and self._color in ds:
            colorarray = ds[self._color].values  # (trajectory, time)

        is_variable_color = colorarray is not None

        if is_variable_color:
            if vmin is None:
                vmin = float(np.nanmin(colorarray))
            if vmax is None:
                vmax = float(np.nanmax(colorarray))
        
        marker_color = (None if is_variable_color
                        else (self._color
                              if (self._color is not None
                                  and not isinstance(self._color, xr.DataArray))
                              else self.DEFAULT_LINE_COLOR))

        sc_kwargs = dict(s=self._markersize, alpha=self._alpha, zorder=10,
                         animated=blit)
        if is_geo:
            sc_kwargs['transform'] = self.gcrs

        x0 = x[:, 0] if x.ndim > 1 else x
        y0 = y[:, 0] if y.ndim > 1 else y

        if is_cartesian:
            ax.set_xlim(np.nanmin(x), np.nanmax(x))
            ax.set_ylim(np.nanmin(y), np.nanmax(y))

        if colorarray is not None:
            sc = ax.scatter(x0, y0, c=colorarray[:, 0],
                            cmap=cmap_obj, vmin=vmin, vmax=vmax, **sc_kwargs)
            if self._colorbar:
                label = self._clabel
                if label is None and isinstance(self._color, str):
                    label = self._color
                fig.colorbar(sc, ax=ax, label=label,
                             orientation='horizontal', pad=0.05, shrink=0.8)
        else:
            sc = ax.scatter(x0, y0, c=marker_color, **sc_kwargs)

        # Static title (renders outside blit cycle, always visible)
        if self._title is not None and self._title != 'auto':
            ax.set_title(self._title)

        # Timestamp shown inside the axes so it works correctly with blitting
        timestamp_text = ax.text(
            0.99, 0.01, '',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize='medium',
            animated=blit, zorder=20,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', pad=4),
        )

        def plot_frame(i):
            xi = x[:, i] if x.ndim > 1 else x
            yi = y[:, i] if y.ndim > 1 else y
            sc.set_offsets(np.column_stack([xi, yi]))

            if colorarray is not None:
                sc.set_array(colorarray[:, i])

            for pm, data_interp in overlay_artists:
                pm.set_array(data_interp.isel(time=i).values.ravel())

            if self._title is not None:
                timestamp_text.set_text(
                    np.datetime_as_string(times[i], unit='s') + ' UTC')

            return [sc, timestamp_text] + [pm for pm, _ in overlay_artists]

        anim = FuncAnimation(fig, plot_frame, frames=frames,
                             interval=1000 // self._fps, blit=blit)
        if blit:
            self._animation = anim
        self._frames = frames
        return anim

    def show(self):
        """
        Display the animation.

        In a Jupyter notebook the animation is embedded as HTML; in a script
        :func:`matplotlib.pyplot.show` is called.

        Returns
        -------
        self
        """
        anim = self.build(blit=False)
        try:
            __IPYTHON__
            from IPython.display import display, HTML
            plt.close(anim._fig)
            display(HTML(anim.to_jshtml()))
        except NameError:
            plt.show()
        return self

    def save(self, filename, fps=None):
        """
        Save the animation to a file.

        Parameters
        ----------
        filename : str or Path
            Output file. Supported formats: ``.gif``, ``.mp4``.
        fps : int, optional
            Frames per second; overrides any prior :meth:`set_fps` call and
            triggers a rebuild.

        Returns
        -------
        self
        """
        filename = str(filename)
        supported = ('.gif', '.mp4')
        if not any(filename.endswith(ext) for ext in supported):
            raise ValueError(
                f"Unsupported format. Use one of {supported}. Got: {filename}")

        if fps is not None:
            self._fps = fps
            self._animation = None  # rebuild with the new fps

        anim = self.build()

        if filename.endswith('.gif'):
            writer = matplotlib.animation.PillowWriter(fps=self._fps)
        else:
            if not matplotlib.animation.FFMpegWriter.isAvailable():
                raise RuntimeError(
                    "FFmpeg is required to save .mp4 files. "
                    "Install it with e.g. `conda install ffmpeg`.")
            writer = matplotlib.animation.FFMpegWriter(
                fps=self._fps,
                codec='libx264',
                bitrate=1800,
                extra_args=[
                    '-profile:v', 'baseline',
                    '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',  # ensure even dimensions
                    '-pix_fmt', 'yuv420p',  # broad player/browser compatibility
                    '-an',  # no audio
                ])

        logger.debug(f'Saving {self._frames} frames to {filename}')
        logger.info(f'Saving animation to {filename}..')

        try:
            from tqdm import tqdm
            pbar = tqdm(total=self._frames, desc=f'Saving {filename}', unit='frame')
            def progress(i, n):
                pbar.update(1)
        except ImportError:
            pbar = None
            progress = None

        anim.save(filename, writer=writer, progress_callback=progress)

        if pbar is not None:
            pbar.close()

        return self

    def _repr_html_(self):
        """Embed the animation inline in a Jupyter notebook."""
        try:
            anim = self.build(blit=False)
            plt.close(anim._fig)
            return anim.to_jshtml()
        except Exception as e:
            logger.error(f'Animation failed: {e}')
            return f'<p>Animation failed: {e}</p>'

