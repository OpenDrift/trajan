import pytest
import xarray as xr
import trajan as _
import matplotlib.pyplot as plt


def test_interpret_sfy(test_data):
    ds = xr.open_dataset(test_data / 'bug32.nc')
    print(ds)

    assert ds.traj.obsdim == 'package'
    assert ds.traj.timedim == 'position_time'

    assert ds.traj.is_2d()


def test_plot_sfy(test_data, plot):
    ds = xr.open_dataset(test_data / 'bug32.nc')
    print(ds)

    ds.traj.plot()

    if plot:
        plt.show()

def test_gridtime(test_data):
    ds = xr.open_dataset(test_data / 'bug32.nc')
    print(ds)

    dg = ds.traj.gridtime('1h')
    print(dg)

@pytest.mark.xfail(reason='timestep methods seems to fail for Traj2d datasets')
def test_timestep(test_data):
    ds = xr.open_dataset(test_data / 'bug32.nc')
    print(ds.traj.timestep())
