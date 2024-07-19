from pathlib import Path
from trajan.readers.omb import read_omb_csv
from trajan.plot.spectra import plot_trajan_spectra
import datetime


def test_plot_spectra_noextraargs(test_data):
    csv_in = test_data / 'csv/omb3.csv'
    xr_data = read_omb_csv(csv_in)
    plot_trajan_spectra(xr_data, plt_show=False)


def test_plot_spectra_withargs(test_data, tmpdir):
    csv_in = test_data / 'csv/omb3.csv'
    xr_data = read_omb_csv(csv_in)
    time_start = datetime.datetime(2022, 6, 16, 12, 30, 0)
    time_end = datetime.datetime(2022, 6, 17, 8, 45, 0)
    vrange_min = -2.0
    vrange_max = 1.5
    fignamesave = tmpdir / 'test.pdf'
    nseconds_gap = 2 * 3600
    plot_trajan_spectra(
        xr_data,
        tuple_date_start_end=(time_start, time_end),
        tuple_vrange_pcolor=(vrange_min, vrange_max),
        plt_show=False,
        fignamesave=fignamesave,
        nseconds_gap=nseconds_gap,
    )
