import pytest
import trajan as ta
import xarray as xr

@pytest.mark.parametrize('sample', ['barents.nc', 'openoil.nc'])
def test_timestep(sample):
    ds = xr.open_dataset(ta.DATA_DIR + sample)
    print(ds.traj.timestep())

