"""
Tools to help plot spectra.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matpltlib.dates as mdates

logger = logging.getLogger(__name__)


# TODO: support type hints
def plot_trajan_spectra(xr_trajan_in, tuple_date_start_end=None, tuple_vrange_pcolor=None, plt_show=True, fignamesave=None, nseconds_gap=6*3600):

    if tuple_date_start_end is not None:
        date_start = tuple_date_start_end[0]
        date_end = tuple_date_start_end[1]
    else:
        # we need a common date start end to be able to share the bottom time axis
        list_date_start = []
        list_date_end = []

        for ind, _ in enumerate(xr_trajan_in["trajectory"]):
            crrt_spectra_times = xr_trajan_in.isel(trajectory=ind)["time_waves_imu"].to_numpy()
            list_date_start.append(np.min(crrt_spectra_times))
            list_date_end.append(np.max(crrt_spectra_times))

        date_start = min(list_date_start)
        date_end = max(list_date_end)

    date_start_md = mdates.date2num(date_start)
    date_end_md = mdates.date2num(date_end)

    logging.log(f"{tuple_date_start_end = }")
    logging.log(f"{date_start_md = }")
    logging.log(f"{date_end_md = }")

    if tuple_vrange_pcolor is None:
        vmin_pcolor = -3.0
        vmax_pcolor = 1.0
    else:
        vmin_pcolor = tuple_vrange_pcolor[0]
        vmax_pcolor = tuple_vrange_pcolor[1]

    logging.log(f"{tuple_vrange_pcolor = }")
    logging.log(f"{vmin_pcolor = }")
    logging.log(f"{vmax_pcolor = }")

    logging.log(f"{plt_show = }")
    logging.log(f"{fignames = }")

    fig, axes = plt.subplots(nrows=len(xr_trajan_in.trajectory), ncols=1)

    spectra_frequencies = xr_trajan_in["frequencies_waves_imu"].to_numpy()

    for ind, crrt_instrument in enumerate(xr_trajan_in["trajectory"]):
        crrt_instrument = str(crrt_instrument.data)
        logging.log(f"{crrt_instrument = }")

        plt.subplot(len(xr_trajan_in.trajectory), 1, ind+1)

        try:
            crrt_spectra = xr_trajan_in.isel(trajectory=ind)["processed_elevation_energy_spectrum"].to_numpy()
            crrt_spectra_times = xr_trajan_in.isel(trajectory=ind)["time_waves_imu"].to_numpy()

            list_datetimes = []
            list_spectra = []

            # avoid streching at the left
            list_datetimes.append(crrt_spectra_times[0] - np.timedelta64(2, 'm'))
            list_spectra.append(np.full(len(spectra_frequencies), np.nan))

            for crrt_spectra_ind in range(1, crrt_spectra.shape[0], 1):
                if np.isnan(crrt_spectra_times[crrt_spectra_ind]):
                    continue

                # if a gap with more than nseconds_gap seconds, fill with NaNs to avoid stretching neighbors over missing data
                seconds_after_previous = float(crrt_spectra_times[crrt_spectra_ind] - crrt_spectra_times[crrt_spectra_ind-1]) / 1e9
                if seconds_after_previous > nseconds_gap:
                    logger.log(f"spectrum index {crrt_spectra_ind} is {seconds_after_previous} seconds after the previous one; insert nan spectra in between to avoid stretching")
                    list_datetimes.append(crrt_spectra_times[crrt_spectra_ind-1] + np.timedelta64(2, 'h'))
                    list_spectra.append(np.full(len(spectra_frequencies), np.nan))
                    list_datetimes.append(crrt_spectra_times[crrt_spectra_ind] - np.timedelta64(2, 'h'))
                    list_spectra.append(np.full(len(spectra_frequencies), np.nan))

                list_spectra.append(crrt_spectra[crrt_spectra_ind, :])
                list_datetimes.append(crrt_spectra_times[crrt_spectra_ind])

            # avoid stretching at the right
            last_datetime = list_datetimes[-1]
            list_datetimes.append(last_datetime + np.timedelta64(2, 'm'))
            list_spectra.append(np.full(len(spectra_frequencies), np.nan))

            pclr = plt.pcolor(list_datetimes, spectra_frequencies, np.log10(np.transpose(np.array(list_spectra))), vmin=vmin_pcolor, vmax=vmax_pcolor)

        except Exception as e:
            logger.error(f"isse with instrument {crrt_instrument}: received exception {e}")

        plt.xlim([date_start_md, date_end_md])
        plt.ylim([0.05, 0.25])

        if ind < len(xr_trajan_in.trajectory)-1:
            plt.xticks([])
        else:
            plt.xticks(rotation=30)

        plt.ylabel("f [Hz]\n({})".format(crrt_instrument), rotation=90, ha='right')

    plt.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(pclr, cax=cbar_ax)
    cbar.set_label('log$_{10}$(S) [m$^2$/Hz]')

    # plt.tight_layout()

    if fignamesave is not None:
        plt.savefig(fignamesave)

    if plt_show:
        plt.show()
