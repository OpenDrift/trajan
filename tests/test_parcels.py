import trajan as _
import xarray as xr
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize('animate', [False, True])
def test_parcels_flatmesh(animate, plot):
    ds = xr.open_dataset('tests/test_data/parcels.zarr', engine='zarr')
    ds = ds.traj.set_crs(None)
    print(ds)
    if animate:
        ds.traj.animate()
    else:
        ds.traj.plot()

    if plot:
        plt.show()
