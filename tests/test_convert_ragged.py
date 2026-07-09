import xarray as xr
import trajan as ta
import matplotlib.pyplot as plt


def test_convert_ragged(plot):
    ds = xr.open_dataset(ta.DATA_DIR + 'xr_spotter_bulk_test_data.nc')

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

def test_convert_ncparticles(plot):
    ds = xr.open_dataset(ta.DATA_DIR + 'gnome_nc_particles.nc')

    print(ds.traj)

    assert ds.traj.ds.trajan_converted_from == 'nc_particles'
