import logging
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import cf_xarray as _

from trajan.accessor import detect_time_dim

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True


class Plot:
    ds: xr.Dataset

    # A lon-lat projection with the currently used globe.
    gcrs = None

    DEFAULT_LINE_COLOR = 'gray'

    def __init__(self, ds):
        self.ds = ds

    def __call__(self, *args, **kwargs):
        if self.ds.attrs['standard_name'] == 'sea_surface_wave_variance_spectral_density':
            return self.spectra(*args, **kwargs)
        else:
            raise ValueError('Unknown wave variable')

    def spectra(self, time, *args, **kwargs):
        """
        Plot the wave spectra information from a trajan compatible xarray.

        Args:

            time: DataArray with times.

            vrange: can be either:
                - None to use the default log range [-3.0, 1.0]
                - a tuple of float to set the log range explicitely

            `nseconds_gap`: float
                Number of seconds between 2 consecutive
                spectra for one instrument above which we consider that there is a
                data loss that should be filled with NaN. This is to avoid "stretching"
                neighboring spectra over long times if an instrument gets offline.

        Returns:

            ax: plt.Axes
        """
        vrange = kwargs.pop('vrange', None)
        nseconds_gap = kwargs.pop('nseconds_gap', 6 * 3600)
        try:
            ax = kwargs.pop('ax')
        except:
            ax = plt.axes()

        if vrange is None:
            vmin_pcolor = -3.0
            vmax_pcolor = 1.0
        else:
            vmin_pcolor = vrange[0]
            vmax_pcolor = vrange[1]

        spectra_frequencies = self.ds.cf['wave_frequency']

        crrt_spectra = self.ds.to_numpy()
        # crrt_spectra_times = detect_time_dim(self.ds, 'obs_waves_imu').to_numpy()
        crrt_spectra_times = time.to_numpy()

        list_datetimes = []
        list_spectra = []

        # avoid streching at the left
        list_datetimes.append(
            crrt_spectra_times[0] - np.timedelta64(2, 'm'))
        list_spectra.append(np.full(len(spectra_frequencies), np.nan))

        for crrt_spectra_ind in range(1, crrt_spectra.shape[0], 1):
            if np.isnan(crrt_spectra_times[crrt_spectra_ind]):
                continue

            # if a gap with more than nseconds_gap seconds, fill with NaNs
            # to avoid stretching neighbors over missing data
            seconds_after_previous = float(
                crrt_spectra_times[crrt_spectra_ind] - crrt_spectra_times[crrt_spectra_ind-1]) / 1e9
            if seconds_after_previous > nseconds_gap:
                logger.debug(
                    f"spectrum index {crrt_spectra_ind} is {seconds_after_previous} seconds \
                    after the previous one; insert nan spectra in between to avoid stretching")
                list_datetimes.append(
                    crrt_spectra_times[crrt_spectra_ind-1] + np.timedelta64(2, 'h'))
                list_spectra.append(
                    np.full(len(spectra_frequencies), np.nan))
                list_datetimes.append(
                    crrt_spectra_times[crrt_spectra_ind] - np.timedelta64(2, 'h'))
                list_spectra.append(
                    np.full(len(spectra_frequencies), np.nan))

            list_spectra.append(crrt_spectra[crrt_spectra_ind, :])
            list_datetimes.append(crrt_spectra_times[crrt_spectra_ind])

        # avoid stretching at the right
        last_datetime = list_datetimes[-1]
        list_datetimes.append(last_datetime + np.timedelta64(2, 'm'))
        list_spectra.append(np.full(len(spectra_frequencies), np.nan))

        pclr = ax.pcolor(list_datetimes, spectra_frequencies, np.log10(
            np.transpose(np.array(list_spectra))), vmin=vmin_pcolor, vmax=vmax_pcolor)

        return ax

