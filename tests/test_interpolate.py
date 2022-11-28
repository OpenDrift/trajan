from numpy.testing import assert_almost_equal
import trajan as ta
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

def test_interpolate_barents(barents, plot):
    """Interpolate subset of drifter time series to 6-hourly means"""

    times = pd.date_range("2022-10-01", "2022-11-01", freq='6H')

    barents_gridded = barents.traj.gridtime(times)
    assert_almost_equal(barents_gridded.isel(trajectory=1).isel(time=100).lon, 19.94633948)

    if plot:
        drifter_names = barents['drifter_names'].data
        barents.traj.plot.set_up_map()  # Large enough to encompass both drifters
        barents.isel(trajectory=1).traj.plot(color='b', label=drifter_names[1])
        barents.isel(trajectory=0).traj.plot(color='r', label=drifter_names[0])

        barents_gridded.isel(trajectory=0).traj.plot(color='r', marker='x')
        barents_gridded.isel(trajectory=1).traj.plot(color='b', marker='x')

        plt.legend()
        plt.show()
