import xarray as xr
import trajan as _
import matplotlib.pyplot as plt


def test_convert_ragged(test_data, plot):
    ds = xr.open_dataset(test_data / 'xr_spotter_bulk_test_data.nc')

    # let us check that the conversion worked well, and that all main functionalities are working
    print(ds.traj)

    # plotting
    ds.traj.plot()

    if plot:
        plt.show()
    else:
        plt.close()

    print("----------------------------------------")
    print("the raw ds:")
    print(ds)
    print("----------------------------------------")

    print("")

    print("----------------------------------------")
    print("the trajanized ds:")
    print(ds.traj.ds)
    print("----------------------------------------")

    assert ds.traj.ds.trajan_converted_from == 'contiguous'

def test_convert_ncparticles(test_data, plot):
    ds = xr.open_dataset(test_data / 'gnome_nc_particles.nc')

    print(ds.traj)

    assert ds.traj.ds.trajan_converted_from == 'nc_particles'
