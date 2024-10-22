import xarray as xr
import trajan as _
import matplotlib.pyplot as plt


def test_convert_ragged(test_data, plot):
    ds = xr.open_dataset(test_data / 'xr_spotter_bulk_test_data.nc')

    # let us check that the conversion worked well, and that all main functionalities are working

    # plotting
    ds.traj.plot()

    if plot:
        plt.show()

    print("----------------------------------------")
    print("the raw ds:")
    print(ds)
    print("----------------------------------------")

    print("")

    print("----------------------------------------")
    print("the trajanized ds:")
    print(ds.traj.ds)
    print("----------------------------------------")
