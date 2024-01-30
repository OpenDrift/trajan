import trajan as _
import xarray as xr
import matplotlib.pyplot as plt
import pytest


def test_parcels_flatmesh(plot):
    ds = xr.open_dataset('tests/test_data/parcels.zarr', engine='zarr')
    ds = ds.traj.set_crs(None)
    print(ds)
    ds.traj.plot()

    if plot:
        plt.show()
    else:
        plt.close()

@pytest.mark.xfail
def test_parcels_flatmesh_animate(plot):
    ds = xr.open_dataset('tests/test_data/parcels.zarr', engine='zarr')
    ds = ds.traj.set_crs(None)
    print(ds)
    ds.traj.animate()

    if plot:
        plt.show()
    else:
        plt.close()
