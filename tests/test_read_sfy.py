import pytest
import xarray as xr
import trajan as ta
import matplotlib.pyplot as plt


def test_interpret_sfy():
    ds = xr.open_dataset(ta.DATA_DIR + 'omb/bug32.nc')
    print(ds)

    assert ds.traj.obs_dim == 'package'
    assert ds.traj.time_varname == 'position_time'

    assert ds.traj.is_ragged()


def test_plot_sfy(plot):
    ds = xr.open_dataset(ta.DATA_DIR + 'omb/bug32.nc')
    print(ds)

    ds.traj.plot()

    if plot:
        plt.show()
    plt.close('all')

def test_gridtime():
    ds = xr.open_dataset(ta.DATA_DIR + 'omb/bug32.nc')
    print(ds)

    dg = ds.traj.gridtime('1h')
    print(dg)

@pytest.mark.xfail(reason='timestep methods seems to fail for Ragged datasets')
def test_timestep():
    ds = xr.open_dataset(ta.DATA_DIR + 'omb/bug32.nc')
    print(ds.traj.timestep())
