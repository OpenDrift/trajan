import xarray as xr
import trajan as _
import matplotlib.pyplot as plt


def test_fastplot(test_data, plot):
    ds = xr.open_dataset(test_data / 'xr_spotter_bulk_test_data.nc')

    # let us check that the conversion worked well, and that all main functionalities are working

    # plotting without centering
    plt.figure()
    ds.traj.plot.fastplot()

    # plotting with centering
    plt.figure()
    ds.traj.plot.fastplot(center_lon_circmean=True)

    if plot:
        plt.show()

